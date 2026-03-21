"""Tests for ChatPipeline — framework-agnostic chat logic."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.chat_pipeline import _LLM_FALLBACK, ChatPipeline, ChatResult
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


def _make_langfuse() -> MagicMock:
    lf = MagicMock()
    lf.create_config = MagicMock(return_value={})
    return lf


def _make_context_builder() -> MagicMock:
    cb = MagicMock()
    cb.assemble_for_llm = MagicMock(return_value=[{"role": "user", "content": "привет"}])
    return cb


def _make_llm_service(response: LLMResponse | None = None) -> AsyncMock:
    svc = AsyncMock()
    svc.generate = AsyncMock(return_value=response or _make_llm_response())
    return svc


def _make_pipeline(
    episode_manager: MagicMock | None = None,
    llm: AsyncMock | None = None,
    context_builder: MagicMock | None = None,
    langfuse: MagicMock | None = None,
    memory: AsyncMock | None = None,
    image_service: AsyncMock | None = None,
    db_client: AsyncMock | None = None,
) -> ChatPipeline:
    return ChatPipeline(
        llm=llm or _make_llm_service(),
        episode_manager=episode_manager or _make_episode_manager(),
        context_builder=context_builder or _make_context_builder(),
        langfuse=langfuse or _make_langfuse(),
        memory=memory,
        image_service=image_service,
        db_client=db_client,
    )


class TestChatPipeline:
    """Tests for ChatPipeline.handle_message()."""

    @pytest.mark.asyncio
    async def test_handle_message_happy_path(self) -> None:
        """User message produces ChatResult with LLM response."""
        llm_resp = _make_llm_response("Привет, как дела?")
        pipeline = _make_pipeline(llm=_make_llm_service(llm_resp))
        result = await pipeline.handle_message(user_id=1, content="привет", user_name="Олег")

        assert isinstance(result, ChatResult)
        assert result.response_text == "Привет, как дела?"
        assert result.llm_response is not None
        assert result.was_rate_limited is False
        assert result.image_bytes is None

    @pytest.mark.asyncio
    async def test_handle_message_llm_failure_returns_fallback(self) -> None:
        """LLM raises -> ChatResult has fallback Russian text."""
        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        pipeline = _make_pipeline(llm=llm_svc)
        result = await pipeline.handle_message(user_id=1, content="привет")

        assert result.response_text == _LLM_FALLBACK
        assert result.llm_response is None
        assert result.was_rate_limited is False

    @pytest.mark.asyncio
    async def test_handle_message_memory_search_failure_still_works(self) -> None:
        """Memory search raises -> LLM still called, response returned."""
        mock_mem = AsyncMock()
        mock_mem.search = AsyncMock(side_effect=RuntimeError("Cognee down"))
        llm_resp = _make_llm_response("Ответ без памяти")
        pipeline = _make_pipeline(llm=_make_llm_service(llm_resp), memory=mock_mem)
        result = await pipeline.handle_message(user_id=1, content="что помнишь?")

        assert result.response_text == "Ответ без памяти"
        assert result.llm_response is not None

    @pytest.mark.asyncio
    async def test_handle_message_rate_limited(self) -> None:
        """Rate limit exceeded -> was_rate_limited=True, no LLM call."""
        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=False)
        pipeline = _make_pipeline(db_client=mock_db)
        result = await pipeline.handle_message(user_id=1, content="ещё")

        assert result.was_rate_limited is True
        assert "лимит" in result.response_text

    @pytest.mark.asyncio
    async def test_handle_message_tool_call_produces_image_bytes(self) -> None:
        """Tool call with send_photo -> image_bytes in ChatResult."""
        tool_calls = [ToolCall(name="send_photo", args={"prompt": "cat"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("Вот твоя картинка!")

        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=[first_resp, final_resp])

        fake_image = b"\x89PNG_fake_image_data"
        mock_img = AsyncMock()
        mock_img.generate = AsyncMock(return_value=fake_image)

        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img)

        with patch(
            "bot.chat_pipeline.SEND_PHOTO_TOOL",
            {"type": "function", "function": {"name": "send_photo"}},
            create=True,
        ):
            from bot.media import image_service as _img_mod

            original = getattr(_img_mod, "SEND_PHOTO_TOOL", None)
            _img_mod.SEND_PHOTO_TOOL = {"type": "function", "function": {"name": "send_photo"}}
            try:
                result = await pipeline.handle_message(user_id=1, content="нарисуй кота")
            finally:
                if original is not None:
                    _img_mod.SEND_PHOTO_TOOL = original

        assert result.image_bytes == fake_image
        assert result.response_text == "Вот твоя картинка!"

    @pytest.mark.asyncio
    async def test_handle_message_no_memory_service(self) -> None:
        """Pipeline works with memory service unavailable."""
        llm_resp = _make_llm_response("Работаю без памяти")
        pipeline = _make_pipeline(llm=_make_llm_service(llm_resp), memory=None)
        result = await pipeline.handle_message(user_id=1, content="привет")

        assert result.response_text == "Работаю без памяти"

    @pytest.mark.asyncio
    async def test_handle_message_no_db_client(self) -> None:
        """Pipeline works without DB client (no rate limit, no usage tracking)."""
        llm_resp = _make_llm_response("Работаю без БД")
        pipeline = _make_pipeline(llm=_make_llm_service(llm_resp), db_client=None)
        result = await pipeline.handle_message(user_id=1, content="привет")

        assert result.response_text == "Работаю без БД"
        assert result.was_rate_limited is False

    @pytest.mark.asyncio
    async def test_handle_message_persists_assistant_message_with_tokens(self) -> None:
        """After handle_message, episode_manager.process_assistant_message called with tokens."""
        llm_resp = LLMResponse(content="Ответ", model="test-model", tokens_in=42, tokens_out=15)
        em = _make_episode_manager()
        pipeline = _make_pipeline(
            episode_manager=em,
            llm=_make_llm_service(llm_resp),
        )
        await pipeline.handle_message(user_id=1, content="привет")

        em.process_assistant_message.assert_called_once_with(
            user_id=1,
            content="Ответ",
            tokens_in=42,
            tokens_out=15,
            model="test-model",
        )


class TestFireAndForget:
    """Tests for the _fire_and_forget background task mechanism."""

    @pytest.mark.asyncio
    async def test_fire_and_forget_logs_exception(self) -> None:
        """Background task that raises -> logger.error called."""
        pipeline = _make_pipeline()

        async def _failing_task() -> None:
            raise ValueError("boom")

        with patch("bot.chat_pipeline.logger") as mock_logger:
            pipeline._fire_and_forget(_failing_task())
            # Let the task complete
            await asyncio.sleep(0.05)

            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "Background task failed" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_fire_and_forget_ignores_cancelled(self) -> None:
        """CancelledError -> no error log."""
        pipeline = _make_pipeline()

        async def _long_task() -> None:
            await asyncio.sleep(100)

        with patch("bot.chat_pipeline.logger") as mock_logger:
            pipeline._fire_and_forget(_long_task())
            # Cancel the task
            for task in pipeline._background_tasks.copy():
                task.cancel()
            await asyncio.sleep(0.05)

            mock_logger.error.assert_not_called()


class TestMemoryWriteCountsIsInstanceState:
    """Verify that memory write counters are per-instance, not global."""

    @pytest.mark.asyncio
    async def test_memory_write_counts_is_instance_state(self) -> None:
        """Two pipelines have independent counters."""
        mem1 = AsyncMock()
        mem1.write_factual = AsyncMock()
        mem1.search = AsyncMock(return_value=[])

        mem2 = AsyncMock()
        mem2.write_factual = AsyncMock()
        mem2.search = AsyncMock(return_value=[])

        p1 = _make_pipeline(memory=mem1)
        p2 = _make_pipeline(memory=mem2)

        # Write to p1's memory background
        await p1._write_memory_background("msg1", user_id=1)
        assert p1._memory_write_counts.get(1, 0) == 1
        assert p2._memory_write_counts.get(1, 0) == 0

        await p2._write_memory_background("msg2", user_id=1)
        assert p2._memory_write_counts.get(1, 0) == 1
        assert p1._memory_write_counts.get(1, 0) == 1
