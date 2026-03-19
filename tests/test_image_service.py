"""Tests for image generation service and handler integration."""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestExtractPhotoPrompt:
    """Tests for the extract_photo_prompt helper."""

    def test_extracts_marker(self) -> None:
        from bot.services.image_service import extract_photo_prompt

        text = "Вот держи! [SEND_PHOTO: cute anime girl selfie smiling]"
        cleaned, prompt = extract_photo_prompt(text)
        assert cleaned == "Вот держи!"
        assert prompt == "cute anime girl selfie smiling"

    def test_no_marker(self) -> None:
        from bot.services.image_service import extract_photo_prompt

        text = "Просто текстовый ответ"
        cleaned, prompt = extract_photo_prompt(text)
        assert cleaned == "Просто текстовый ответ"
        assert prompt is None

    def test_empty_text(self) -> None:
        from bot.services.image_service import extract_photo_prompt

        cleaned, prompt = extract_photo_prompt("")
        assert cleaned == ""
        assert prompt is None

    def test_marker_only(self) -> None:
        from bot.services.image_service import extract_photo_prompt

        text = "[SEND_PHOTO: a photo of sunset]"
        cleaned, prompt = extract_photo_prompt(text)
        assert cleaned == ""
        assert prompt == "a photo of sunset"


class TestImageServiceGenerate:
    """Tests for ImageService.generate() method."""

    @pytest.fixture
    def mock_openai_client(self) -> AsyncMock:
        client = AsyncMock()
        mock_response = MagicMock()
        mock_image_data = MagicMock()
        mock_image_data.b64_json = base64.b64encode(b"fake_png_data").decode()
        mock_response.data = [mock_image_data]
        client.images.generate = AsyncMock(return_value=mock_response)
        return client

    @pytest.fixture
    def service(self, mock_openai_client: AsyncMock) -> Any:
        from bot.services.image_service import ImageService

        svc = ImageService.__new__(ImageService)
        svc._client = mock_openai_client
        svc._model = "gpt-image-1"
        svc._send_counts: dict[int, dict[str, int]] = {}
        return svc

    @pytest.mark.asyncio
    async def test_generate_happy_path(self, service: Any, mock_openai_client: AsyncMock) -> None:
        result = await service.generate("cute cat", user_id=123)
        assert result is not None
        assert isinstance(result, bytes)
        mock_openai_client.images.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_rate_limited(self, service: Any) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        service._send_counts[123] = {today: 5}
        result = await service.generate("test", user_id=123)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_empty_prompt(self, service: Any) -> None:
        result = await service.generate("", user_id=123)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_api_error_returns_none(
        self, service: Any, mock_openai_client: AsyncMock
    ) -> None:
        mock_openai_client.images.generate = AsyncMock(side_effect=Exception("API error"))
        result = await service.generate("test", user_id=123)
        assert result is None

    @pytest.mark.asyncio
    async def test_unavailable_returns_none(self) -> None:
        from bot.services.image_service import ImageService

        svc = ImageService.__new__(ImageService)
        svc._client = None
        svc._send_counts: dict[int, dict[str, int]] = {}
        result = await svc.generate("test", user_id=123)
        assert result is None


class TestHandlerImageIntegration:
    """Tests for image handling in chat handler."""

    @pytest.mark.asyncio
    async def test_chat_with_photo_marker_sends_image(self) -> None:
        """When LLM includes [SEND_PHOTO:], image generated + sent via answer_photo."""
        from bot.services.llm_service import LLMResponse

        mock_msg = MagicMock()
        mock_msg.from_user = MagicMock(id=123, first_name="Test")
        mock_msg.text = "пришли фото"
        mock_msg.answer = AsyncMock()
        mock_msg.answer_photo = AsyncMock()
        mock_msg.photo = None
        mock_msg.document = None
        mock_msg.voice = None
        mock_msg.video = None
        mock_msg.audio = None
        mock_msg.sticker = None
        mock_msg.location = None
        mock_msg.contact = None
        mock_msg.caption = None

        llm_resp = LLMResponse(
            content="Вот держи! [SEND_PHOTO: cute selfie photo]",
            model="test",
            tokens_in=10,
            tokens_out=20,
        )

        mock_episode_mgr = AsyncMock()
        mock_episode_mgr.process_user_message = AsyncMock(
            return_value=MagicMock(
                episode=MagicMock(id="ep1"),
                is_new_episode=False,
                switch_decision=MagicMock(reason="same topic"),
            )
        )
        mock_episode_mgr.process_assistant_message = AsyncMock()
        mock_episode_mgr.get_recent_messages = AsyncMock(return_value=[])

        mock_image_svc = MagicMock()
        mock_image_svc.generate = AsyncMock(return_value=b"fake_image_bytes")

        with (
            patch("bot.handlers.get_episode_manager_service", return_value=mock_episode_mgr),
            patch("bot.handlers.get_llm_service") as mock_llm,
            patch("bot.handlers.get_image_service", return_value=mock_image_svc),
            patch("bot.handlers.IMAGE_SERVICE_AVAILABLE", True),
            patch("bot.handlers.MEMORY_SERVICE_AVAILABLE", False),
            patch("bot.handlers.DB_CLIENT_AVAILABLE", False),
        ):
            mock_llm.return_value.generate = AsyncMock(return_value=llm_resp)

            from bot.handlers import chat

            await chat(mock_msg)

        mock_msg.answer_photo.assert_called_once()
        # Text should be cleaned (no marker)
        if mock_msg.answer.called:
            call_text = mock_msg.answer.call_args[0][0]
            assert "[SEND_PHOTO" not in call_text

    @pytest.mark.asyncio
    async def test_chat_without_marker_sends_text_only(self) -> None:
        """When LLM has no marker, only text is sent."""
        from bot.services.llm_service import LLMResponse

        mock_msg = MagicMock()
        mock_msg.from_user = MagicMock(id=123, first_name="Test")
        mock_msg.text = "привет"
        mock_msg.answer = AsyncMock()
        mock_msg.answer_photo = AsyncMock()
        mock_msg.photo = None
        mock_msg.document = None
        mock_msg.voice = None
        mock_msg.video = None
        mock_msg.audio = None
        mock_msg.sticker = None
        mock_msg.location = None
        mock_msg.contact = None
        mock_msg.caption = None

        llm_resp = LLMResponse(
            content="Привет! Как дела?", model="test", tokens_in=5, tokens_out=10
        )

        mock_episode_mgr = AsyncMock()
        mock_episode_mgr.process_user_message = AsyncMock(
            return_value=MagicMock(
                episode=MagicMock(id="ep1"),
                is_new_episode=False,
                switch_decision=MagicMock(reason="same topic"),
            )
        )
        mock_episode_mgr.process_assistant_message = AsyncMock()
        mock_episode_mgr.get_recent_messages = AsyncMock(return_value=[])

        with (
            patch("bot.handlers.get_episode_manager_service", return_value=mock_episode_mgr),
            patch("bot.handlers.get_llm_service") as mock_llm,
            patch("bot.handlers.IMAGE_SERVICE_AVAILABLE", True),
            patch("bot.handlers.MEMORY_SERVICE_AVAILABLE", False),
            patch("bot.handlers.DB_CLIENT_AVAILABLE", False),
        ):
            mock_llm.return_value.generate = AsyncMock(return_value=llm_resp)

            from bot.handlers import chat

            await chat(mock_msg)

        mock_msg.answer.assert_called_once_with("Привет! Как дела?")
        mock_msg.answer_photo.assert_not_called()
