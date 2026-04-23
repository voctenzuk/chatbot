"""Image generation service via OpenAI-compatible API (OpenRouter).

Generates images from text prompts with optional reference image for
identity preservation. Returns ImageResult with cost metadata.

Architecture (tool calling):
    LLM receives send_photo / send_sprite tool definitions
           |
           v
    LLM returns tool_call: send_photo(prompt="...") or send_sprite(emotion="...")
           |
           v
    ImageService.generate(prompt) or ImageService.get_sprite(emotion)
           |                                |
           v                                v
    chat.completions.create()          Supabase Storage (cached)
    modalities=["image"]
           |
           v
    base64 data URL -> bytes -> bot.send_photo()
"""

import base64
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from bot.character import SPRITE_EMOTIONS
from bot.config import settings

if TYPE_CHECKING:
    from bot.character import CharacterConfig

try:
    from openai import AsyncOpenAI

    _openai_available = True
except ImportError:
    _openai_available = False
    AsyncOpenAI = None

# Note: Image generation is tracked via DB-based rate limiting (try_consume_photo),
# not via in-memory counters.

# Default cost per image in cents (SeeDream 4.5 on OpenRouter)
DEFAULT_IMAGE_COST_CENTS = 4.0


@dataclass(frozen=True)
class ImageResult:
    """Result of image generation with cost metadata."""

    image_bytes: bytes
    cost_cents: float
    provider: str


# Tool schema for LangChain bind_tools() -- bare dict format (no "type"/"function" wrapper)
SEND_PHOTO_TOOL = {
    "name": "send_photo",
    "description": (
        "Send a photo to the user. Use sparingly and contextually: "
        "selfies, food photos, scenery, mood photos. "
        "Describe the image in English for the image generator."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Image description in English for the image generator",
            }
        },
        "required": ["prompt"],
    },
}

SEND_SPRITE_TOOL = {
    "name": "send_sprite",
    "description": (
        "Send a quick emotion photo to the user. Use for reactions, "
        "emotions, greetings. Available emotions: "
        + ", ".join(SPRITE_EMOTIONS)
        + ". Instant delivery, no generation delay."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": list(SPRITE_EMOTIONS),
                "description": "The emotion to express",
            }
        },
        "required": ["emotion"],
    },
}

# Regex to extract base64 data from data URL (e.g. "data:image/png;base64,iVBOR...")
_DATA_URL_RE = re.compile(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)")


class ImageService:
    """Generates images via OpenAI-compatible API (OpenRouter).

    Uses chat.completions.create() with modalities=["image"] for generation.
    Supports reference images for identity preservation via multimodal messages.
    Gracefully unavailable if openai package is not installed or API key is not configured.
    """

    def __init__(self, character: "CharacterConfig | None" = None) -> None:
        self._character = character
        self._model = settings.image_model
        self._sprite_cache: dict[str, bytes] = {}

        if not _openai_available:
            self._client = None
            logger.warning("ImageService: openai package not installed")
            return

        if not settings.image_api_key:
            self._client = None
            logger.warning("ImageService: IMAGE_API_KEY not configured")
            return

        kwargs: dict[str, Any] = {"api_key": settings.image_api_key}
        if settings.image_base_url:
            kwargs["base_url"] = settings.image_base_url

        self._client = AsyncOpenAI(**kwargs)  # type: ignore[misc]
        logger.info("ImageService initialized with model={}", self._model)

    @property
    def available(self) -> bool:
        """Whether the service can generate images."""
        return self._client is not None

    async def generate(self, prompt: str, user_id: int) -> ImageResult | None:
        """Generate an image from a text prompt, optionally with reference image.

        Flow:
        1. If character has reference_image_url → multimodal request (identity preservation)
        2. If reference fails → fallback to text-only with appearance_en prefix
        3. If that fails → return None

        Returns:
            ImageResult with bytes and cost metadata, or None on failure.
        """
        if not self.available:
            logger.debug("ImageService unavailable -- skipping generation")
            return None

        if not prompt or not prompt.strip():
            return None

        # Path 1: Reference image (identity preservation)
        ref_url = self._character.reference_image_url if self._character is not None else None
        if ref_url:
            try:
                result = await self._generate_with_reference(prompt, ref_url, user_id)
                if result is not None:
                    return result
            except Exception as exc:
                logger.warning(
                    "Reference image generation failed for user {}, falling back to text-only: {}",
                    user_id,
                    exc,
                )

        # Path 2: Text-only with appearance prefix
        return await self._generate_text_only(prompt, user_id)

    async def get_sprite(self, emotion: str) -> bytes | None:
        """Get pre-generated sprite by emotion name. Instant, $0.

        Downloads from Supabase Storage on first call, caches in memory.
        Returns None if character has no sprites or emotion not found.
        """
        if self._character is None or self._character.sprite_urls is None:
            return None

        url = self._character.sprite_urls.get(emotion)
        if not url:
            return None

        # Return from cache if available
        if emotion in self._sprite_cache:
            return self._sprite_cache[emotion]

        # Download and cache
        image_bytes = await self._download_image(url)
        if image_bytes is not None:
            self._sprite_cache[emotion] = image_bytes
        return image_bytes

    async def _generate_with_reference(
        self,
        prompt: str,
        reference_url: str,
        user_id: int,
    ) -> ImageResult | None:
        """Generate image with reference image for identity preservation."""
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": reference_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        return await self._call_chat_completions(messages, user_id, "reference")

    async def _generate_text_only(
        self,
        prompt: str,
        user_id: int,
    ) -> ImageResult | None:
        """Generate image from text prompt only, with appearance prefix."""
        if self._character is not None:
            prompt = f"{self._character.appearance_en}. Scene: {prompt}"

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        return await self._call_chat_completions(messages, user_id, "text-only")

    async def _call_chat_completions(
        self,
        messages: list[dict[str, Any]],
        user_id: int,
        mode: str,
    ) -> ImageResult | None:
        """Call chat.completions.create() with modalities=["image"] and extract result."""
        assert self._client is not None

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                modalities=["image"],  # type: ignore[arg-type]
            )

            # Extract base64 image from response content
            if not response.choices:
                logger.warning("Image API returned no choices for user {} ({})", user_id, mode)
                return None

            content = response.choices[0].message.content
            image_bytes = self._extract_image_bytes(content)
            if image_bytes is None:
                logger.warning("No image data in response for user {} ({})", user_id, mode)
                return None

            logger.info(
                "Generated image for user {} ({} bytes, mode={}, prompt={})",
                user_id,
                len(image_bytes),
                mode,
                str(messages[-1].get("content", ""))[:50],
            )
            return ImageResult(
                image_bytes=image_bytes,
                cost_cents=DEFAULT_IMAGE_COST_CENTS,
                provider=self._model,
            )

        except Exception as exc:
            logger.warning("Image generation failed for user {} ({}): {}", user_id, mode, exc)
            return None

    @staticmethod
    def _extract_image_bytes(content: Any) -> bytes | None:
        """Extract image bytes from chat completion response content.

        Handles both:
        - String content with embedded base64 data URL
        - List content blocks with image_url type
        """
        if content is None:
            return None

        def _decode_match(match: re.Match[str]) -> bytes | None:
            try:
                return base64.b64decode(match.group(1))
            except Exception:
                return None

        # Content is a string — look for data URL
        if isinstance(content, str):
            match = _DATA_URL_RE.search(content)
            if match:
                return _decode_match(match)
            return None

        # Content is a list of content blocks
        if isinstance(content, list):
            for block in content:
                if hasattr(block, "type") and block.type == "image_url":
                    url = getattr(block, "image_url", None)
                    if url is not None:
                        url_str = getattr(url, "url", url) if not isinstance(url, str) else url
                        match = _DATA_URL_RE.search(str(url_str))
                        if match:
                            return _decode_match(match)

        return None

    async def _download_image(self, url: str) -> bytes | None:
        """Download image bytes from a URL (used for sprites)."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except Exception as exc:
            logger.warning("Image download failed for {}: {}", url[:80], exc)
            return None


# ---------------------------------------------------------------------------
# Dependency-injection helpers
# ---------------------------------------------------------------------------

_image_service: ImageService | None = None


def get_image_service() -> ImageService:
    """Get or create the global ImageService singleton."""
    global _image_service
    if _image_service is None:
        _image_service = ImageService()
    return _image_service


def set_image_service(service: ImageService | None) -> None:
    """Set the global ImageService instance (useful for testing)."""
    global _image_service
    _image_service = service
