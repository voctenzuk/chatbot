"""Tests for LLM integration in chat handlers.

All external services are mocked: episode_manager, memory_service,
llm_service, context_builder.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiogram.types import Chat, User

from bot.services.llm_service import LLMResponse


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

    async def answer(self, text: str, **kwargs):
        self._last_answer = text
        return MagicMock()


def _make_message_result(episode_id: str | None = None):
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
def mock_episode_manager():
    mgr = AsyncMock()
    mgr.process_user_message = AsyncMock(return_value=_make_message_result())
    mgr.process_assistant_message = AsyncMock(return_value=_make_message_result())
    mgr.get_recent_messages = AsyncMock(return_value=[])
    return mgr


@pytest.fixture
def mock_memory_service():
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    return svc


@pytest.fixture
def mock_llm_service():
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
def mock_context_builder():
    builder = MagicMock()
    builder.assemble_for_llm = MagicMock(
        return_value=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ]
    )
    return builder


@pytest.fixture
def patched_handlers(
    mock_episode_manager,
    mock_memory_service,
    mock_llm_service,
    mock_context_builder,
):
    """Patch all service accessors used in handlers."""
    with (
        patch(
            "bot.handlers.get_episode_manager_service",
            return_value=mock_episode_manager,
        ),
        patch(
            "bot.handlers.get_memory_service",
            return_value=mock_memory_service,
        ),
        patch(
            "bot.handlers.get_llm_service",
            return_value=mock_llm_service,
        ),
        patch(
            "bot.handlers.get_context_builder",
            return_value=mock_context_builder,
        ),
        patch(
            "bot.handlers.get_system_prompt",
            return_value="You are a helpful assistant.",
        ),
    ):
        yield {
            "episode_manager": mock_episode_manager,
            "memory_service": mock_memory_service,
            "llm_service": mock_llm_service,
            "context_builder": mock_context_builder,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChatLLMIntegration:
    """Tests for LLM-powered chat handler."""

    @pytest.mark.asyncio
    async def test_chat_calls_llm_and_returns_response(self, patched_handlers):
        """Full flow: user sends text -> gets LLM response (not stub)."""
        from bot.handlers import chat

        msg = MockMessage(text="How are you?", user_id=42)
        await chat(msg)

        patched_handlers["llm_service"].generate.assert_called_once()
        assert msg._last_answer == "LLM reply text"

    @pytest.mark.asyncio
    async def test_chat_memory_search_failure_still_responds(self, patched_handlers):
        """Memory search raises -> handler still calls LLM with empty memories."""
        from bot.handlers import chat

        patched_handlers["memory_service"].search = AsyncMock(
            side_effect=RuntimeError("memory down")
        )

        msg = MockMessage(text="tell me something", user_id=42)
        await chat(msg)

        # LLM was still called
        patched_handlers["llm_service"].generate.assert_called_once()
        assert msg._last_answer == "LLM reply text"

    @pytest.mark.asyncio
    async def test_chat_llm_failure_returns_fallback(self, patched_handlers):
        """LLM raises -> handler returns graceful fallback in Russian."""
        from bot.handlers import chat

        patched_handlers["llm_service"].generate = AsyncMock(
            side_effect=RuntimeError("LLM exploded")
        )

        msg = MockMessage(text="hello", user_id=42)
        await chat(msg)

        assert msg._last_answer is not None
        assert "Прости" in msg._last_answer or "не получается" in msg._last_answer

    @pytest.mark.asyncio
    async def test_chat_persists_assistant_message_with_tokens(self, patched_handlers):
        """process_assistant_message receives tokens from LLMResponse."""
        from bot.handlers import chat

        msg = MockMessage(text="hi", user_id=42)
        await chat(msg)

        call_kwargs = patched_handlers["episode_manager"].process_assistant_message.call_args.kwargs
        assert call_kwargs["tokens_in"] == 15
        assert call_kwargs["tokens_out"] == 8
        assert call_kwargs["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_chat_includes_system_prompt(self, patched_handlers):
        """Assembled messages include system prompt via context_builder."""
        from bot.handlers import chat

        msg = MockMessage(text="hey", user_id=42)
        await chat(msg)

        cb = patched_handlers["context_builder"]
        call_kwargs = cb.assemble_for_llm.call_args.kwargs
        assert "system_prompt" in call_kwargs
        assert call_kwargs["system_prompt"] == "You are a helpful assistant."


class TestStartHandlerUnchanged:
    """Verify /start still works as before."""

    @pytest.mark.asyncio
    async def test_start_handler_unchanged(self, patched_handlers):
        """Start handler still returns the greeting."""
        from bot.handlers import start

        msg = MockMessage(text="/start", user_id=42)
        await start(msg)

        assert msg._last_answer is not None
        assert "рядом" in msg._last_answer or "Привет" in msg._last_answer
