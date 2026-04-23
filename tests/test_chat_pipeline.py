"""Tests for ChatPipeline — framework-agnostic chat logic."""

import asyncio
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.chat_pipeline import (
    LLM_FALLBACK,
    ChatPipeline,
    ChatResult,
)
from bot.config import settings
from bot.llm.service import LLMResponse, ToolCall


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
    lf.trace = MagicMock(return_value=nullcontext())
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
        assert result.image_bytes == []

    @pytest.mark.asyncio
    async def test_handle_message_llm_failure_returns_fallback(self) -> None:
        """LLM raises -> ChatResult has fallback Russian text."""
        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        pipeline = _make_pipeline(llm=llm_svc)
        result = await pipeline.handle_message(user_id=1, content="привет")

        assert result.response_text == LLM_FALLBACK
        assert result.llm_response is None
        assert result.was_rate_limited is False

    @pytest.mark.asyncio
    async def test_handle_message_memory_search_failure_still_works(self) -> None:
        """Memory search raises -> LLM still called, response returned."""
        mock_mem = AsyncMock()
        mock_mem.search = AsyncMock(side_effect=RuntimeError("Memory unavailable"))
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
        from bot.media.image_service import ImageResult

        tool_calls = [ToolCall(name="send_photo", args={"prompt": "cat"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("Вот твоя картинка!")

        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=[first_resp, final_resp])

        fake_image = ImageResult(
            image_bytes=b"\x89PNG_fake_image_data", cost_cents=4.0, provider="t"
        )
        mock_img = AsyncMock()
        mock_img.generate = AsyncMock(return_value=fake_image)

        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.try_consume_photo = AsyncMock(return_value=True)
        mock_db.increment_usage = AsyncMock()

        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img, db_client=mock_db)

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

        assert b"\x89PNG_fake_image_data" in result.image_bytes
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


class TestCostTrackingAndPhotoRateLimit:
    """Tests for cost tracking via increment_usage and photo rate limiting."""

    @pytest.mark.asyncio
    async def test_cost_cents_calculated_and_passed_to_increment_usage(self) -> None:
        """cost_cents is calculated from tokens and passed to db_client.increment_usage."""
        llm_resp = LLMResponse(content="Ответ", model="test", tokens_in=1000, tokens_out=500)
        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.increment_usage = AsyncMock()

        pipeline = _make_pipeline(llm=_make_llm_service(llm_resp), db_client=mock_db)
        await pipeline.handle_message(user_id=1, content="привет")

        mock_db.increment_usage.assert_called_once()
        call_kwargs = mock_db.increment_usage.call_args
        cost_cents = call_kwargs.kwargs.get("cost_cents") or call_kwargs[1].get("cost_cents")
        if cost_cents is None:
            # Positional args: (user_id, msg_count=, tokens_in=, tokens_out=, cost_cents=)
            cost_cents = call_kwargs[0][4] if len(call_kwargs[0]) > 4 else None
        assert cost_cents is not None
        expected = (
            1000 * settings.cost_per_1m_input + 500 * settings.cost_per_1m_output
        ) / 1_000_000
        assert abs(cost_cents - expected) < 1e-12

    @pytest.mark.asyncio
    async def test_cost_cents_includes_image_cost_when_image_generated(self) -> None:
        """cost_cents uses ImageResult.cost_cents when an image is generated."""
        from bot.media.image_service import ImageResult

        tool_calls = [ToolCall(name="send_photo", args={"prompt": "cat"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("Вот картинка!")

        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=[first_resp, final_resp])

        fake_result = ImageResult(image_bytes=b"\x89PNG_fake", cost_cents=4.0, provider="test")
        mock_img = AsyncMock()
        mock_img.generate = AsyncMock(return_value=fake_result)

        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.try_consume_photo = AsyncMock(return_value=True)
        mock_db.increment_usage = AsyncMock()

        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img, db_client=mock_db)
        result = await pipeline.handle_message(user_id=1, content="нарисуй кота")

        assert len(result.image_bytes) > 0
        mock_db.increment_usage.assert_called_once()
        call_kwargs = mock_db.increment_usage.call_args
        cost_cents = call_kwargs.kwargs.get("cost_cents") or call_kwargs[1].get("cost_cents")
        if cost_cents is None:
            cost_cents = call_kwargs[0][4] if len(call_kwargs[0]) > 4 else None
        assert cost_cents is not None
        assert cost_cents >= 4.0

    @pytest.mark.asyncio
    async def test_photo_rate_limit_returns_limit_message(self) -> None:
        """try_consume_photo returns False -> tool returns rate limit message."""
        tool_calls = [ToolCall(name="send_photo", args={"prompt": "cat"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("Не получилось.")

        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=[first_resp, final_resp])

        mock_img = AsyncMock()
        mock_img.generate = AsyncMock(return_value=b"should_not_be_called")

        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.try_consume_photo = AsyncMock(return_value=False)
        mock_db.increment_usage = AsyncMock()

        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img, db_client=mock_db)
        result = await pipeline.handle_message(user_id=1, content="фото")

        # Image should NOT have been generated
        mock_img.generate.assert_not_called()
        # No image bytes delivered
        assert result.image_bytes == []

    @pytest.mark.asyncio
    async def test_photo_no_db_returns_unavailable(self) -> None:
        """db_client is None -> tool returns 'Image service unavailable'."""
        tool_calls = [ToolCall(name="send_photo", args={"prompt": "cat"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("Сервис недоступен.")

        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=[first_resp, final_resp])

        mock_img = AsyncMock()
        mock_img.generate = AsyncMock(return_value=b"nope")

        # No db_client at all
        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img, db_client=None)
        result = await pipeline.handle_message(user_id=1, content="фото")

        mock_img.generate.assert_not_called()
        assert result.image_bytes == []

    @pytest.mark.asyncio
    async def test_photo_rate_limit_true_generates_normally(self) -> None:
        """try_consume_photo returns True -> photo generated and returned normally."""
        from bot.media.image_service import ImageResult

        tool_calls = [ToolCall(name="send_photo", args={"prompt": "selfie"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("Вот!")

        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=[first_resp, final_resp])

        fake_result = ImageResult(image_bytes=b"\x89PNG_data", cost_cents=4.0, provider="test")
        mock_img = AsyncMock()
        mock_img.generate = AsyncMock(return_value=fake_result)

        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.try_consume_photo = AsyncMock(return_value=True)
        mock_db.increment_usage = AsyncMock()

        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img, db_client=mock_db)
        result = await pipeline.handle_message(user_id=1, content="селфи")

        mock_img.generate.assert_called_once()
        assert b"\x89PNG_data" in result.image_bytes


class TestSpriteToolPipeline:
    """Tests for send_sprite tool handling in ChatPipeline."""

    @pytest.mark.asyncio
    async def test_send_sprite_returns_image(self) -> None:
        """send_sprite tool call produces sprite bytes in result."""
        tool_calls = [ToolCall(name="send_sprite", args={"emotion": "smile"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("Улыбаюсь!")

        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=[first_resp, final_resp])

        mock_img = AsyncMock()
        mock_img.get_sprite = AsyncMock(return_value=b"smile-png")

        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.increment_usage = AsyncMock()

        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img, db_client=mock_db)
        result = await pipeline.handle_message(user_id=1, content="улыбнись")

        mock_img.get_sprite.assert_called_once_with("smile")
        assert b"smile-png" in result.image_bytes

    @pytest.mark.asyncio
    async def test_send_sprite_not_rate_limited(self) -> None:
        """send_sprite does NOT call try_consume_photo."""
        tool_calls = [ToolCall(name="send_sprite", args={"emotion": "sad"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("Грустно...")

        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=[first_resp, final_resp])

        mock_img = AsyncMock()
        mock_img.get_sprite = AsyncMock(return_value=b"sad-png")

        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.try_consume_photo = AsyncMock(return_value=True)
        mock_db.increment_usage = AsyncMock()

        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img, db_client=mock_db)
        await pipeline.handle_message(user_id=1, content="грустно")

        mock_db.try_consume_photo.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_sprite_cost_is_zero(self) -> None:
        """Sprite images have zero cost in usage tracking."""
        tool_calls = [ToolCall(name="send_sprite", args={"emotion": "wink"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("😉")

        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=[first_resp, final_resp])

        mock_img = AsyncMock()
        mock_img.get_sprite = AsyncMock(return_value=b"wink-png")

        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.increment_usage = AsyncMock()

        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img, db_client=mock_db)
        await pipeline.handle_message(user_id=1, content="подмигни")

        mock_db.increment_usage.assert_called_once()
        call_kwargs = mock_db.increment_usage.call_args
        cost_cents = call_kwargs.kwargs.get("cost_cents") or call_kwargs[1].get("cost_cents")
        if cost_cents is None:
            cost_cents = call_kwargs[0][4] if len(call_kwargs[0]) > 4 else None
        # Cost should be token cost only, no image cost
        assert cost_cents is not None
        assert cost_cents < 1.0  # Only token cost, no image cost added

    @pytest.mark.asyncio
    async def test_send_sprite_unavailable(self) -> None:
        """When sprite is unavailable, no image in result."""
        tool_calls = [ToolCall(name="send_sprite", args={"emotion": "angry"}, id="tc-1")]
        first_resp = _make_llm_response("", tool_calls=tool_calls)
        final_resp = _make_llm_response("Не получилось.")

        llm_svc = AsyncMock()
        llm_svc.generate = AsyncMock(side_effect=[first_resp, final_resp])

        mock_img = AsyncMock()
        mock_img.get_sprite = AsyncMock(return_value=None)

        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.increment_usage = AsyncMock()

        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img, db_client=mock_db)
        result = await pipeline.handle_message(user_id=1, content="злись")

        assert result.image_bytes == []

    @pytest.mark.asyncio
    async def test_tools_list_includes_both_tools(self) -> None:
        """When image service available, both send_photo and send_sprite are passed to LLM."""
        llm_svc = AsyncMock()
        llm_resp = _make_llm_response("Привет!")
        llm_svc.generate = AsyncMock(return_value=llm_resp)

        mock_img = AsyncMock()

        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.increment_usage = AsyncMock()

        pipeline = _make_pipeline(llm=llm_svc, image_service=mock_img, db_client=mock_db)
        await pipeline.handle_message(user_id=1, content="привет")

        # Check tools passed to LLM
        call_kwargs = llm_svc.generate.call_args
        tools = call_kwargs.kwargs.get("tools") or (
            call_kwargs[1].get("tools") if len(call_kwargs) > 1 else None
        )
        assert tools is not None
        tool_names = [t["name"] for t in tools]
        assert "send_photo" in tool_names
        assert "send_sprite" in tool_names


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


class TestWriteMemoryBackground:
    """Verify _write_memory_background calls write_factual."""

    @pytest.mark.asyncio
    async def test_write_memory_background_calls_write_factual(self) -> None:
        """_write_memory_background delegates to write_factual with correct args."""
        mem = AsyncMock()
        mem.write_factual = AsyncMock()
        mem.search = AsyncMock(return_value=[])

        pipeline = _make_pipeline(memory=mem)
        await pipeline._write_memory_background("msg1", user_id=1)

        mem.write_factual.assert_called_once_with(content="msg1", user_id=1)
