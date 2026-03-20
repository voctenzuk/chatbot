"""Tests for image generation service and tool-calling handler integration."""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSendPhotoToolSchema:
    """Tests for the SEND_PHOTO_TOOL schema."""

    def test_tool_schema_structure(self) -> None:
        from bot.services.image_service import SEND_PHOTO_TOOL

        assert SEND_PHOTO_TOOL["name"] == "send_photo"
        assert "prompt" in SEND_PHOTO_TOOL["parameters"]["properties"]
        assert "prompt" in SEND_PHOTO_TOOL["parameters"]["required"]

    def test_send_photo_tool_format(self) -> None:
        """SEND_PHOTO_TOOL should be in bare dict format for bind_tools()."""
        from bot.services.image_service import SEND_PHOTO_TOOL

        assert "name" in SEND_PHOTO_TOOL
        assert "parameters" in SEND_PHOTO_TOOL
        assert SEND_PHOTO_TOOL["name"] == "send_photo"
        # Should NOT have the wrapped OpenAI format
        assert "function" not in SEND_PHOTO_TOOL
        assert SEND_PHOTO_TOOL.get("type") != "function"


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


class TestHandlerToolCallIntegration:
    """Tests for tool-call-based image handling in chat handler."""

    @pytest.mark.asyncio
    async def test_chat_with_tool_call_sends_image(self) -> None:
        """When LLM returns send_photo tool_call, image is generated and sent."""
        from bot.services.llm_service import LLMResponse, ToolCall

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
            content="Вот держи!",
            model="test",
            tokens_in=10,
            tokens_out=20,
            tool_calls=[ToolCall(name="send_photo", args={"prompt": "cute selfie"}, id="t1")],
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

        mock_llm_svc = AsyncMock()
        mock_llm_svc.generate = AsyncMock(return_value=llm_resp)

        with (
            patch("bot.handlers.get_episode_manager_service", return_value=mock_episode_mgr),
            patch("bot.chat_pipeline.get_llm_service", return_value=mock_llm_svc),
            patch("bot.chat_pipeline.get_image_service", return_value=mock_image_svc),
            patch("bot.chat_pipeline.IMAGE_SERVICE_AVAILABLE", True),
            patch("bot.chat_pipeline.MEMORY_SERVICE_AVAILABLE", False),
            patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", False),
            patch("bot.handlers.DB_CLIENT_AVAILABLE", False),
            patch(
                "bot.chat_pipeline.get_langfuse_service",
                return_value=MagicMock(create_config=MagicMock(return_value={})),
            ),
            patch(
                "bot.chat_pipeline.get_context_builder",
                return_value=MagicMock(
                    assemble_for_llm=MagicMock(
                        return_value=[
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": "пришли фото"},
                        ]
                    )
                ),
            ),
            patch("bot.chat_pipeline.get_system_prompt", return_value="sys"),
        ):
            from bot.handlers import chat

            await chat(mock_msg)

        mock_msg.answer_photo.assert_called_once()
        # generate() called once in tool loop; image cached and reused for delivery
        assert mock_image_svc.generate.call_count == 1
        mock_image_svc.generate.assert_called_once_with("cute selfie", 123)

    @pytest.mark.asyncio
    async def test_chat_without_tool_call_sends_text_only(self) -> None:
        """When LLM has no tool_calls, only text is sent."""
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
            content="Привет! Как дела?",
            model="test",
            tokens_in=5,
            tokens_out=10,
            tool_calls=[],
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

        mock_llm_svc = AsyncMock()
        mock_llm_svc.generate = AsyncMock(return_value=llm_resp)

        with (
            patch("bot.handlers.get_episode_manager_service", return_value=mock_episode_mgr),
            patch("bot.chat_pipeline.get_llm_service", return_value=mock_llm_svc),
            patch("bot.chat_pipeline.IMAGE_SERVICE_AVAILABLE", True),
            patch("bot.chat_pipeline.MEMORY_SERVICE_AVAILABLE", False),
            patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", False),
            patch("bot.handlers.DB_CLIENT_AVAILABLE", False),
            patch(
                "bot.chat_pipeline.get_langfuse_service",
                return_value=MagicMock(create_config=MagicMock(return_value={})),
            ),
            patch(
                "bot.chat_pipeline.get_context_builder",
                return_value=MagicMock(
                    assemble_for_llm=MagicMock(
                        return_value=[
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": "привет"},
                        ]
                    )
                ),
            ),
            patch("bot.chat_pipeline.get_system_prompt", return_value="sys"),
        ):
            from bot.handlers import chat

            await chat(mock_msg)

        mock_msg.answer.assert_called_once_with("Привет! Как дела?")
        mock_msg.answer_photo.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_image_gen_failure_sends_text(self) -> None:
        """When image generation fails, text response is still sent."""
        from bot.services.llm_service import LLMResponse, ToolCall

        mock_msg = MagicMock()
        mock_msg.from_user = MagicMock(id=123, first_name="Test")
        mock_msg.text = "фото"
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
            content="Вот!",
            model="test",
            tokens_in=5,
            tokens_out=10,
            tool_calls=[ToolCall(name="send_photo", args={"prompt": "test"}, id="t1")],
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
        mock_image_svc.generate = AsyncMock(return_value=None)  # generation failed

        mock_llm_svc = AsyncMock()
        mock_llm_svc.generate = AsyncMock(return_value=llm_resp)

        with (
            patch("bot.handlers.get_episode_manager_service", return_value=mock_episode_mgr),
            patch("bot.chat_pipeline.get_llm_service", return_value=mock_llm_svc),
            patch("bot.chat_pipeline.get_image_service", return_value=mock_image_svc),
            patch("bot.chat_pipeline.IMAGE_SERVICE_AVAILABLE", True),
            patch("bot.chat_pipeline.MEMORY_SERVICE_AVAILABLE", False),
            patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", False),
            patch("bot.handlers.DB_CLIENT_AVAILABLE", False),
            patch(
                "bot.chat_pipeline.get_langfuse_service",
                return_value=MagicMock(create_config=MagicMock(return_value={})),
            ),
            patch(
                "bot.chat_pipeline.get_context_builder",
                return_value=MagicMock(
                    assemble_for_llm=MagicMock(
                        return_value=[
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": "фото"},
                        ]
                    )
                ),
            ),
            patch("bot.chat_pipeline.get_system_prompt", return_value="sys"),
        ):
            from bot.handlers import chat

            await chat(mock_msg)

        # Photo not sent, but text fallback is
        mock_msg.answer_photo.assert_not_called()
        mock_msg.answer.assert_called_once_with("Вот!")
