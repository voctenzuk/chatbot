"""Tests for ChatPipeline — framework-agnostic chat logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.chat_pipeline import ChatPipeline, ChatResult, _LLM_FALLBACK
from bot.services.llm_service import LLMResponse, ToolCall


def _make_episode_manager() -> MagicMock:
    """Create a mock EpisodeManager with standard async methods."""
    em = MagicMock()
    em.process_user_message = AsyncMock(
        return_value=MagicMock(
            episode=MagicMock(id="ep-1"),
            is_new_episode=False,
            switch_decision=MagicMock(reason="keep"),
        )
    )
    em.process_assistant_message = AsyncMock()
    em.get_current_episode = AsyncMock(return_value=MagicMock(id="ep-1"))
    em.get_recent_messages = AsyncMock(return_value=[])
    return em


def _make_llm_response(content: str = "Привет!", **kwargs: object) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        tokens_in=10,
        tokens_out=20,
        **kwargs,  # type: ignore[arg-type]
    )


def _make_pipeline(episode_manager: MagicMock | None = None) -> ChatPipeline:
    return ChatPipeline(episode_manager=episode_manager or _make_episode_manager())


class TestChatPipeline:
    """Tests for ChatPipeline.handle_message()."""

    @pytest.fixture(autouse=True)
    def _reset_memory_counts(self) -> None:
        """Reset module-level memory write counters between tests."""
        from bot.chat_pipeline import _memory_write_counts

        _memory_write_counts.clear()

    @pytest.mark.asyncio
    @patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", False)
    @patch("bot.chat_pipeline.get_langfuse_service")
    @patch("bot.chat_pipeline.get_llm_service")
    @patch("bot.chat_pipeline.get_context_builder")
    async def test_handle_message_happy_path(
        self,
        mock_ctx_builder: MagicMock,
        mock_llm_svc: MagicMock,
        mock_langfuse: MagicMock,
    ) -> None:
        """User message produces ChatResult with LLM response."""
        mock_langfuse.return_value.create_config.return_value = {}
        mock_ctx_builder.return_value.assemble_for_llm.return_value = [
            {"role": "user", "content": "привет"}
        ]
        llm_resp = _make_llm_response("Привет, как дела?")
        mock_llm_svc.return_value.generate = AsyncMock(return_value=llm_resp)

        pipeline = _make_pipeline()
        result = await pipeline.handle_message(user_id=1, content="привет", user_name="Олег")

        assert isinstance(result, ChatResult)
        assert result.response_text == "Привет, как дела?"
        assert result.llm_response is not None
        assert result.was_rate_limited is False
        assert result.image_bytes is None

    @pytest.mark.asyncio
    @patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", False)
    @patch("bot.chat_pipeline.get_langfuse_service")
    @patch("bot.chat_pipeline.get_llm_service")
    @patch("bot.chat_pipeline.get_context_builder")
    async def test_handle_message_llm_failure_returns_fallback(
        self,
        mock_ctx_builder: MagicMock,
        mock_llm_svc: MagicMock,
        mock_langfuse: MagicMock,
    ) -> None:
        """LLM raises → ChatResult has fallback Russian text."""
        mock_langfuse.return_value.create_config.return_value = {}
        mock_ctx_builder.return_value.assemble_for_llm.return_value = []
        mock_llm_svc.return_value.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

        pipeline = _make_pipeline()
        result = await pipeline.handle_message(user_id=1, content="привет")

        assert result.response_text == _LLM_FALLBACK
        assert result.llm_response is None
        assert result.was_rate_limited is False

    @pytest.mark.asyncio
    @patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", False)
    @patch("bot.chat_pipeline.get_memory_service")
    @patch("bot.chat_pipeline.MEMORY_SERVICE_AVAILABLE", True)
    @patch("bot.chat_pipeline.get_langfuse_service")
    @patch("bot.chat_pipeline.get_llm_service")
    @patch("bot.chat_pipeline.get_context_builder")
    async def test_handle_message_memory_search_failure_still_works(
        self,
        mock_ctx_builder: MagicMock,
        mock_llm_svc: MagicMock,
        mock_langfuse: MagicMock,
        mock_mem_svc: MagicMock,
    ) -> None:
        """Memory search raises → LLM still called, response returned."""
        mock_langfuse.return_value.create_config.return_value = {}
        mock_ctx_builder.return_value.assemble_for_llm.return_value = []
        mock_mem_svc.return_value.search = AsyncMock(side_effect=RuntimeError("Cognee down"))
        llm_resp = _make_llm_response("Ответ без памяти")
        mock_llm_svc.return_value.generate = AsyncMock(return_value=llm_resp)

        pipeline = _make_pipeline()
        result = await pipeline.handle_message(user_id=1, content="что помнишь?")

        assert result.response_text == "Ответ без памяти"
        assert result.llm_response is not None

    @pytest.mark.asyncio
    @patch("bot.chat_pipeline.get_db_client")
    @patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", True)
    async def test_handle_message_rate_limited(
        self,
        mock_db: MagicMock,
    ) -> None:
        """Rate limit exceeded → was_rate_limited=True, no LLM call."""
        mock_db.return_value.check_rate_limit = AsyncMock(return_value=False)

        pipeline = _make_pipeline()
        result = await pipeline.handle_message(user_id=1, content="ещё")

        assert result.was_rate_limited is True
        assert "лимит" in result.response_text

    @pytest.mark.asyncio
    @patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", False)
    @patch("bot.chat_pipeline.get_image_service")
    @patch("bot.chat_pipeline.IMAGE_SERVICE_AVAILABLE", True)
    @patch(
        "bot.chat_pipeline.SEND_PHOTO_TOOL",
        {"type": "function", "function": {"name": "send_photo"}},
    )
    @patch("bot.chat_pipeline.get_langfuse_service")
    @patch("bot.chat_pipeline.get_llm_service")
    @patch("bot.chat_pipeline.get_context_builder")
    async def test_handle_message_tool_call_produces_image_bytes(
        self,
        mock_ctx_builder: MagicMock,
        mock_llm_svc: MagicMock,
        mock_langfuse: MagicMock,
        mock_img_svc: MagicMock,
    ) -> None:
        """Tool call with send_photo → image_bytes in ChatResult."""
        mock_langfuse.return_value.create_config.return_value = {}
        mock_ctx_builder.return_value.assemble_for_llm.return_value = []

        tool_calls = [ToolCall(name="send_photo", args={"prompt": "cat"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("Вот твоя картинка!")
        mock_llm_svc.return_value.generate = AsyncMock(side_effect=[first_resp, final_resp])

        fake_image = b"\x89PNG_fake_image_data"
        mock_img_svc.return_value.generate = AsyncMock(return_value=fake_image)

        pipeline = _make_pipeline()
        result = await pipeline.handle_message(user_id=1, content="нарисуй кота")

        assert result.image_bytes == fake_image
        assert result.response_text == "Вот твоя картинка!"

    @pytest.mark.asyncio
    @patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", False)
    @patch("bot.chat_pipeline.MEMORY_SERVICE_AVAILABLE", False)
    @patch("bot.chat_pipeline.get_langfuse_service")
    @patch("bot.chat_pipeline.get_llm_service")
    @patch("bot.chat_pipeline.get_context_builder")
    async def test_handle_message_no_memory_service(
        self,
        mock_ctx_builder: MagicMock,
        mock_llm_svc: MagicMock,
        mock_langfuse: MagicMock,
    ) -> None:
        """Pipeline works with memory service unavailable."""
        mock_langfuse.return_value.create_config.return_value = {}
        mock_ctx_builder.return_value.assemble_for_llm.return_value = []
        llm_resp = _make_llm_response("Работаю без памяти")
        mock_llm_svc.return_value.generate = AsyncMock(return_value=llm_resp)

        pipeline = _make_pipeline()
        result = await pipeline.handle_message(user_id=1, content="привет")

        assert result.response_text == "Работаю без памяти"

    @pytest.mark.asyncio
    @patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", False)
    @patch("bot.chat_pipeline.get_langfuse_service")
    @patch("bot.chat_pipeline.get_llm_service")
    @patch("bot.chat_pipeline.get_context_builder")
    async def test_handle_message_no_db_client(
        self,
        mock_ctx_builder: MagicMock,
        mock_llm_svc: MagicMock,
        mock_langfuse: MagicMock,
    ) -> None:
        """Pipeline works without DB client (no rate limit, no usage tracking)."""
        mock_langfuse.return_value.create_config.return_value = {}
        mock_ctx_builder.return_value.assemble_for_llm.return_value = []
        llm_resp = _make_llm_response("Работаю без БД")
        mock_llm_svc.return_value.generate = AsyncMock(return_value=llm_resp)

        pipeline = _make_pipeline()
        result = await pipeline.handle_message(user_id=1, content="привет")

        assert result.response_text == "Работаю без БД"
        assert result.was_rate_limited is False

    @pytest.mark.asyncio
    @patch("bot.chat_pipeline.DB_CLIENT_AVAILABLE", False)
    @patch("bot.chat_pipeline.get_langfuse_service")
    @patch("bot.chat_pipeline.get_llm_service")
    @patch("bot.chat_pipeline.get_context_builder")
    async def test_handle_message_persists_assistant_message_with_tokens(
        self,
        mock_ctx_builder: MagicMock,
        mock_llm_svc: MagicMock,
        mock_langfuse: MagicMock,
    ) -> None:
        """After handle_message, episode_manager.process_assistant_message called with tokens."""
        mock_langfuse.return_value.create_config.return_value = {}
        mock_ctx_builder.return_value.assemble_for_llm.return_value = []
        llm_resp = LLMResponse(content="Ответ", model="test-model", tokens_in=42, tokens_out=15)
        mock_llm_svc.return_value.generate = AsyncMock(return_value=llm_resp)

        em = _make_episode_manager()
        pipeline = _make_pipeline(episode_manager=em)
        await pipeline.handle_message(user_id=1, content="привет")

        em.process_assistant_message.assert_called_once_with(
            user_id=1,
            content="Ответ",
            tokens_in=42,
            tokens_out=15,
            model="test-model",
        )
