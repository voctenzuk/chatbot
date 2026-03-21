"""Image generation service using OpenAI Images API.

Generates images from text prompts and returns raw bytes for sending
via Telegram bot.send_photo(). Uses gpt-image-1 by default.

Architecture (tool calling):
    LLM receives send_photo tool definition
           |
           v
    LLM returns tool_call: send_photo(prompt="...")
           |
           v
    ImageService.generate(prompt)
           |
           v
    OpenAI Images API (gpt-image-1)
           |
           v
    base64 -> bytes -> bot.send_photo()
"""

import base64
from typing import TYPE_CHECKING

from loguru import logger

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


class ImageService:
    """Generates images via OpenAI Images API.

    Gracefully unavailable if openai package is not installed
    or API key is not configured.
    """

    def __init__(self, character: "CharacterConfig | None" = None) -> None:
        self._character = character
        self._model = settings.image_model

        if not _openai_available:
            self._client = None
            logger.warning("ImageService: openai package not installed")
            return

        if not settings.image_api_key:
            self._client = None
            logger.warning("ImageService: IMAGE_API_KEY not configured")
            return

        kwargs: dict = {"api_key": settings.image_api_key}
        if settings.image_base_url:
            kwargs["base_url"] = settings.image_base_url

        self._client = AsyncOpenAI(**kwargs)  # type: ignore[misc]
        logger.info("ImageService initialized with model={}", self._model)

    @property
    def available(self) -> bool:
        """Whether the service can generate images."""
        return self._client is not None

    async def generate(self, prompt: str, user_id: int) -> bytes | None:
        """Generate an image from a text prompt.

        Args:
            prompt: Description of the image to generate.
            user_id: Telegram user ID (for logging).

        Returns:
            Image bytes if successful, None if unavailable or failed.
        """
        if not self.available:
            logger.debug("ImageService unavailable -- skipping generation")
            return None

        if not prompt or not prompt.strip():
            return None

        # Prepend character appearance for visual consistency
        if self._character is not None:
            prompt = f"{self._character.appearance_en}. Scene: {prompt}"

        try:
            result = await self._client.images.generate(  # type: ignore[union-attr]
                model=self._model,
                prompt=prompt,
                n=1,
                size="1024x1024",
                response_format="b64_json",
            )

            if result.data and result.data[0].b64_json:
                image_bytes = base64.b64decode(result.data[0].b64_json)
                logger.info(
                    "Generated image for user {} ({} bytes, prompt: {})",
                    user_id,
                    len(image_bytes),
                    prompt[:50],
                )
                return image_bytes

            logger.warning("OpenAI returned empty image data")
            return None

        except Exception as exc:
            logger.warning("Image generation failed for user {}: {}", user_id, exc)
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
