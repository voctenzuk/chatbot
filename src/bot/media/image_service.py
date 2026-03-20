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

from __future__ import annotations

import base64
from datetime import datetime

from loguru import logger

from bot.config import settings

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None  # type: ignore[assignment, misc]

_MAX_IMAGES_PER_DAY = 5

# Note: Image generation is tracked via rate limiting (_send_counts),
# not via Langfuse (no LangChain call to instrument).

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

    Rate-limited to _MAX_IMAGES_PER_DAY per user per day.
    Gracefully unavailable if openai package is not installed
    or API key is not configured.
    """

    def __init__(self) -> None:
        self._send_counts: dict[int, dict[str, int]] = {}
        self._model = settings.image_model

        if not OPENAI_AVAILABLE:
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

    def _check_rate_limit(self, user_id: int) -> bool:
        """Return True if user can receive more images today."""
        today = datetime.now().strftime("%Y-%m-%d")
        user_counts = self._send_counts.get(user_id, {})
        return user_counts.get(today, 0) < _MAX_IMAGES_PER_DAY

    def _record_send(self, user_id: int) -> None:
        """Record that an image was generated for user."""
        today = datetime.now().strftime("%Y-%m-%d")
        if user_id not in self._send_counts:
            self._send_counts[user_id] = {}
        self._send_counts[user_id][today] = self._send_counts[user_id].get(today, 0) + 1
        self._send_counts[user_id] = {
            k: v for k, v in self._send_counts[user_id].items() if k == today
        }

    async def generate(self, prompt: str, user_id: int) -> bytes | None:
        """Generate an image from a text prompt.

        Args:
            prompt: Description of the image to generate.
            user_id: Telegram user ID (for rate limiting).

        Returns:
            Image bytes if successful, None if unavailable/rate-limited/failed.
        """
        if not self.available:
            logger.debug("ImageService unavailable -- skipping generation")
            return None

        if not prompt or not prompt.strip():
            return None

        if not self._check_rate_limit(user_id):
            logger.debug("Image rate limit reached for user {}", user_id)
            return None

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
                self._record_send(user_id)
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
