"""Tests for Vision Service."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from bot.services.vision_service import (
    VisionService,
    ImageDescription,
    get_vision_service,
    set_vision_service,
)


class TestImageDescription:
    """Tests for ImageDescription dataclass."""

    def test_create_description(self):
        """Test creating image description."""
        desc = ImageDescription(
            description="A beautiful sunset",
            tags=["sunset", "nature"],
            mood="peaceful",
            style="realistic",
        )

        assert desc.description == "A beautiful sunset"
        assert desc.tags == ["sunset", "nature"]
        assert desc.mood == "peaceful"
        assert desc.style == "realistic"


class TestVisionService:
    """Tests for VisionService."""

    @pytest.fixture
    def service(self):
        """Create fresh service instance with mocked client."""
        service = VisionService(model="test-vision-model", api_key="test-key")
        service.client = MagicMock(spec=httpx.AsyncClient)
        return service

    async def test_describe_image_with_bytes(self, service):
        """Test describing image from bytes."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "Фото заката на пляже"}}]
        })
        service.client.post = AsyncMock(return_value=mock_response)

        image_bytes = b"fake_image_data"
        result = await service.describe_image(image_bytes, user_id=123)

        assert isinstance(result, ImageDescription)
        assert result.description == "Фото заката на пляже"
        service.client.post.assert_called_once()

    async def test_describe_image_with_base64(self, service):
        """Test describing image from base64 string."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "Портрет девушки"}}]
        })
        service.client.post = AsyncMock(return_value=mock_response)

        result = await service.describe_image("base64string", user_id=456)

        assert result.description == "Портрет девушки"

    async def test_describe_generated_image(self, service):
        """Test describing AI-generated image."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "Красивый пейзаж"}}]
        })
        service.client.post = AsyncMock(return_value=mock_response)

        result = await service.describe_generated_image(
            prompt="красивый пейзаж",
            image_data=b"fake_data",
            user_id=789,
        )

        assert isinstance(result, ImageDescription)
        # Check that context includes the prompt
        call_args = service.client.post.call_args
        messages = call_args[1]["json"]["messages"]
        assert "сгенерировано по запросу" in messages[0]["content"]

    def test_extract_tags(self, service):
        """Test tag extraction from description."""
        desc = "Это фотография портрета девушки на фоне заката"
        tags = service._extract_tags(desc)

        assert "портрет" in tags
        assert "закат" in tags

    async def test_close(self, service):
        """Test closing the client."""
        service.client.aclose = AsyncMock()
        await service.close()
        service.client.aclose.assert_called_once()


class TestVisionServiceGlobal:
    """Tests for global vision service instance."""

    def test_get_vision_service_creates_instance(self):
        """Test that get_vision_service creates default instance."""
        set_vision_service(None)  # Reset

        with patch("bot.services.vision_service.settings") as mock_settings:
            mock_settings.vision_model = "x-ai/grok-2-vision-1212"
            mock_settings.llm_api_key = "test-key"
            mock_settings.llm_base_url = "https://openrouter.ai/api/v1"

            service = get_vision_service()
            assert isinstance(service, VisionService)

    def test_set_vision_service(self):
        """Test setting global instance."""
        custom_service = VisionService(model="custom", api_key="test")
        set_vision_service(custom_service)

        retrieved = get_vision_service()
        assert retrieved is custom_service
