"""Tests for LLM integration in chat handlers.

All external services are mocked: episode_manager, memory_service,
llm_service, context_builder.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiogram.types import Chat, User

from bot.chat_pipeline import ChatPipeline
from bot.llm.service import LLMResponse, ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockMessage:
    """Minimal Telegram message mock for handler tests."""

    text: str | None
    caption: str | None
    from_user: Any
    chat: Any
    photo: Any
    document: Any
    voice: Any
    video: Any
    audio: Any
    sticker: Any
    location: Any
    contact: Any

    def __init__(
        self,
        text: str | None = None,
        caption: str | None = None,
        user_id: int = 12345,
        first_name: str | None = None,
    ):
        self.text = text
        self.caption = caption
        self.from_user = MagicMock(spec=User)
        self.from_user.id = user_id
        self.from_user.first_name = first_name
        self.chat = MagicMock(spec=Chat)
        self.chat.id = user_id
        self.photo = None
        self.document = None
        self.voice = None
        self.video = None
        self.audio = None
        self.sticker = None
        self.location = None
        self.contact = None
        self._last_answer: str | None = None

    async def answer(self, text: str, **kwargs: Any) -> MagicMock:
        self._last_answer = text
        return MagicMock()


def _make_message_result(episode_id: str | None = None) -> MagicMock:
    """Build a mock MessageResult for episode_manager."""
    mock_episode = MagicMock()
    mock_episode.id = episode_id or str(uuid4())

    mock_msg = MagicMock()
    mock_msg.id = str(uuid4())
    mock_msg.episode_id = mock_episode.id

    mock_decision = MagicMock()
    mock_decision.should_switch = False
    mock_decision.reason = "Continuing"
    mock_decision.confidence = 0.5
    mock_decision.trigger_type = None

    result = MagicMock()
    result.message = mock_msg
    result.episode = mock_episode
    result.is_new_episode = False
    result.switch_decision = mock_decision
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_episode_manager() -> AsyncMock:
    mgr = AsyncMock()
    mgr.process_user_message = AsyncMock(return_value=_make_message_result())
    mgr.process_assistant_message = AsyncMock(return_value=_make_message_result())
    mgr.get_recent_messages = AsyncMock(return_value=[])
    mgr.get_current_episode = AsyncMock(return_value=None)
    return mgr


@pytest.fixture
def mock_memory_service() -> AsyncMock:
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    return svc


@pytest.fixture
def mock_llm_service() -> AsyncMock:
    svc = AsyncMock()
    svc.generate = AsyncMock(
        return_value=LLMResponse(
            content="LLM reply text",
            model="test-model",
            tokens_in=15,
            tokens_out=8,
        )
    )
    return svc


@pytest.fixture
def mock_context_builder() -> MagicMock:
    builder = MagicMock()
    builder.assemble_for_llm = MagicMock(
        return_value=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ]
    )
    return builder


@pytest.fixture
def mock_langfuse_service() -> MagicMock:
    svc = MagicMock()
    svc.create_config = MagicMock(return_value={})
    return svc


@pytest.fixture
def pipeline(
    mock_episode_manager: AsyncMock,
    mock_memory_service: AsyncMock,
    mock_llm_service: AsyncMock,
    mock_context_builder: MagicMock,
    mock_langfuse_service: MagicMock,
) -> ChatPipeline:
    """Create a ChatPipeline with all mock dependencies."""
    with patch(
        "bot.chat_pipeline.get_system_prompt",
        return_value="You are a helpful assistant.",
    ):
        return ChatPipeline(
            llm=mock_llm_service,
            episode_manager=mock_episode_manager,
            context_builder=mock_context_builder,
            langfuse=mock_langfuse_service,
            memory=mock_memory_service,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChatLLMIntegration:
    """Tests for LLM-powered chat handler."""

    @pytest.mark.asyncio
    async def test_chat_calls_llm_and_returns_response(self, pipeline: ChatPipeline) -> None:
        """Full flow: user sends text -> gets LLM response (not stub)."""
        from bot.handlers import chat

        msg = MockMessage(text="How are you?", user_id=42)
        with patch(
            "bot.chat_pipeline.get_system_prompt",
            return_value="You are a helpful assistant.",
        ):
            await chat(msg, pipeline=pipeline)  # type: ignore[arg-type]

        pipeline._llm.generate.assert_called_once()
        assert msg._last_answer == "LLM reply text"

    @pytest.mark.asyncio
    async def test_chat_memory_search_failure_still_responds(self, pipeline: ChatPipeline) -> None:
        """Memory search raises -> handler still calls LLM with empty memories."""
        from bot.handlers import chat

        pipeline._memory.search = AsyncMock(side_effect=RuntimeError("memory down"))

        msg = MockMessage(text="tell me something", user_id=42)
        with patch(
            "bot.chat_pipeline.get_system_prompt",
            return_value="You are a helpful assistant.",
        ):
            await chat(msg, pipeline=pipeline)  # type: ignore[arg-type]

        pipeline._llm.generate.assert_called_once()
        assert msg._last_answer == "LLM reply text"

    @pytest.mark.asyncio
    async def test_chat_llm_failure_returns_fallback(self, pipeline: ChatPipeline) -> None:
        """LLM raises -> handler returns graceful fallback in Russian."""
        from bot.handlers import chat

        pipeline._llm.generate = AsyncMock(side_effect=RuntimeError("LLM exploded"))

        msg = MockMessage(text="hello", user_id=42)
        with patch(
            "bot.chat_pipeline.get_system_prompt",
            return_value="You are a helpful assistant.",
        ):
            await chat(msg, pipeline=pipeline)  # type: ignore[arg-type]

        assert msg._last_answer is not None
        assert "Прости" in msg._last_answer or "не получается" in msg._last_answer

    @pytest.mark.asyncio
    async def test_chat_persists_assistant_message_with_tokens(
        self, pipeline: ChatPipeline
    ) -> None:
        """process_assistant_message receives tokens from LLMResponse."""
        from bot.handlers import chat

        msg = MockMessage(text="hi", user_id=42)
        with patch(
            "bot.chat_pipeline.get_system_prompt",
            return_value="You are a helpful assistant.",
        ):
            await chat(msg, pipeline=pipeline)  # type: ignore[arg-type]

        call_kwargs = pipeline._episode_manager.process_assistant_message.call_args.kwargs
        assert call_kwargs["tokens_in"] == 15
        assert call_kwargs["tokens_out"] == 8
        assert call_kwargs["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_chat_includes_system_prompt(self, pipeline: ChatPipeline) -> None:
        """Assembled messages include system prompt via context_builder."""
        from bot.handlers import chat

        msg = MockMessage(text="hey", user_id=42)
        with patch(
            "bot.chat_pipeline.get_system_prompt",
            return_value="You are a helpful assistant.",
        ):
            await chat(msg, pipeline=pipeline)  # type: ignore[arg-type]

        call_kwargs = pipeline._context_builder.assemble_for_llm.call_args.kwargs
        assert "system_prompt" in call_kwargs
        assert call_kwargs["system_prompt"].startswith("You are a helpful assistant.")


class TestToolExecutionLoop:
    """Tests for the tool execution loop inside ChatPipeline._generate_llm_response."""

    @pytest.mark.asyncio
    async def test_tool_loop_calls_llm_twice_and_returns_final_text(
        self,
        mock_episode_manager: AsyncMock,
        mock_context_builder: MagicMock,
        mock_langfuse_service: MagicMock,
        mock_memory_service: AsyncMock,
    ) -> None:
        """When LLM returns tool_calls, _generate_llm_response calls LLM a second time
        and returns the final text from that second call."""
        tool_call = ToolCall(name="send_photo", args={"prompt": "a sunset"}, id="tc_001")

        first_response = LLMResponse(
            content="",
            model="test-model",
            tokens_in=10,
            tokens_out=5,
            tool_calls=[tool_call],
        )
        second_response = LLMResponse(
            content="Here is your sunset photo!",
            model="test-model",
            tokens_in=20,
            tokens_out=12,
            tool_calls=[],
        )

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=[first_response, second_response])

        mock_img = AsyncMock()
        mock_img.generate = AsyncMock(return_value=b"fake_image_bytes")

        with (
            patch(
                "bot.media.image_service.SEND_PHOTO_TOOL",
                {"name": "send_photo"},
            ),
            patch(
                "bot.chat_pipeline.get_system_prompt",
                return_value="You are a helpful assistant.",
            ),
        ):
            p = ChatPipeline(
                llm=mock_llm,
                episode_manager=mock_episode_manager,
                context_builder=mock_context_builder,
                langfuse=mock_langfuse_service,
                memory=mock_memory_service,
                image_service=mock_img,
            )
            result = await p._generate_llm_response(
                user_id=42,
                content="send me a sunset photo",
                user_name="Sasha",
            )

        assert mock_llm.generate.call_count == 2
        assert result.content == "Here is your sunset photo!"
        assert result.tokens_in == 30
        assert result.tokens_out == 17
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "send_photo"

    @pytest.mark.asyncio
    async def test_tool_loop_second_call_has_tool_message(
        self,
        mock_episode_manager: AsyncMock,
        mock_context_builder: MagicMock,
        mock_langfuse_service: MagicMock,
        mock_memory_service: AsyncMock,
    ) -> None:
        """The second LLM call must include a tool-role message in its conversation."""
        tool_call = ToolCall(name="send_photo", args={"prompt": "a cat"}, id="tc_002")

        first_response = LLMResponse(
            content="",
            model="test-model",
            tokens_in=10,
            tokens_out=5,
            tool_calls=[tool_call],
        )
        second_response = LLMResponse(
            content="Here is your cat!",
            model="test-model",
            tokens_in=18,
            tokens_out=6,
            tool_calls=[],
        )

        mock_llm = AsyncMock()
        generate_mock = AsyncMock(side_effect=[first_response, second_response])
        mock_llm.generate = generate_mock

        mock_img = AsyncMock()
        mock_img.generate = AsyncMock(return_value=b"img")

        with (
            patch(
                "bot.media.image_service.SEND_PHOTO_TOOL",
                {"name": "send_photo"},
            ),
            patch(
                "bot.chat_pipeline.get_system_prompt",
                return_value="You are a helpful assistant.",
            ),
        ):
            p = ChatPipeline(
                llm=mock_llm,
                episode_manager=mock_episode_manager,
                context_builder=mock_context_builder,
                langfuse=mock_langfuse_service,
                memory=mock_memory_service,
                image_service=mock_img,
            )
            await p._generate_llm_response(
                user_id=99,
                content="show me a cat",
                user_name=None,
            )

        second_call_args = generate_mock.call_args_list[1]
        follow_up_messages: list[dict[str, Any]] = second_call_args.args[0]

        tool_role_messages = [m for m in follow_up_messages if m.get("role") == "tool"]
        assert len(tool_role_messages) == 1
        assert tool_role_messages[0]["tool_call_id"] == "tc_002"
        assert "Photo generated" in tool_role_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_tool_loop_skipped_when_no_tool_calls(
        self,
        mock_episode_manager: AsyncMock,
        mock_context_builder: MagicMock,
        mock_langfuse_service: MagicMock,
        mock_memory_service: AsyncMock,
    ) -> None:
        """When the first LLM response has no tool_calls, LLM is called only once."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="plain text reply",
                model="test-model",
                tokens_in=5,
                tokens_out=3,
                tool_calls=[],
            )
        )

        with patch(
            "bot.chat_pipeline.get_system_prompt",
            return_value="You are a helpful assistant.",
        ):
            p = ChatPipeline(
                llm=mock_llm,
                episode_manager=mock_episode_manager,
                context_builder=mock_context_builder,
                langfuse=mock_langfuse_service,
                memory=mock_memory_service,
            )
            result = await p._generate_llm_response(
                user_id=7,
                content="hello",
                user_name=None,
            )

        assert mock_llm.generate.call_count == 1
        assert result.content == "plain text reply"


class TestStartHandlerUnchanged:
    """Verify /start still works as before."""

    @pytest.mark.asyncio
    async def test_start_handler_unchanged(
        self,
        mock_episode_manager: AsyncMock,
        mock_langfuse_service: MagicMock,
        mock_context_builder: MagicMock,
        mock_llm_service: AsyncMock,
    ) -> None:
        """Start handler still returns the greeting."""
        from bot.handlers import start

        mock_pipeline = MagicMock(spec=ChatPipeline)
        mock_pipeline._episode_manager = mock_episode_manager
        mock_pipeline._db_client = None

        msg = MockMessage(text="/start", user_id=42)
        await start(msg, pipeline=mock_pipeline)  # type: ignore[arg-type]

        assert msg._last_answer is not None
        assert "рядом" in msg._last_answer or "Привет" in msg._last_answer
