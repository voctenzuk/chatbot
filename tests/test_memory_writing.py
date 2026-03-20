"""Tests for memory writing in chat handler.

After LLM responds, the conversation turn (user message + bot reply) is
written to the memory service via a background task (_write_memory_background).
Cognify is triggered periodically after N writes.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiogram.types import Chat, User

from bot.services.llm_service import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _drain_tasks() -> None:
    """Yield control so that pending ``asyncio.create_task`` tasks complete.

    Multiple iterations handle nested tasks (e.g. ``_write_memory_background``
    spawns ``_run_cognify_background`` via another ``create_task``).
    """
    for _ in range(5):
        await asyncio.sleep(0)


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
        self.from_user.username = "testuser"
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
        self.successful_payment = None
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
    return mgr


@pytest.fixture
def mock_memory_service() -> AsyncMock:
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    svc.write_factual = AsyncMock(return_value="mem_123")
    svc.cognify = AsyncMock()
    return svc


@pytest.fixture
def mock_llm_service() -> AsyncMock:
    svc = AsyncMock()
    svc.generate = AsyncMock(
        return_value=LLMResponse(
            content="test reply",
            model="test-model",
            tokens_in=10,
            tokens_out=20,
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
def mock_db_client() -> AsyncMock:
    db = AsyncMock()
    db.check_rate_limit = AsyncMock(return_value=True)
    db.increment_usage = AsyncMock()
    db.provision_user = AsyncMock()
    return db


@pytest.fixture
def patched_handlers(
    mock_episode_manager: AsyncMock,
    mock_memory_service: AsyncMock,
    mock_llm_service: AsyncMock,
    mock_context_builder: MagicMock,
    mock_db_client: AsyncMock,
) -> Any:
    """Patch all service accessors used in handlers and chat_pipeline."""
    mock_langfuse_service = MagicMock()
    mock_langfuse_service.create_config = MagicMock(return_value={})
    mock_episode_manager.get_current_episode = AsyncMock(return_value=None)
    with (
        patch(
            "bot.handlers.get_episode_manager_service",
            new=AsyncMock(return_value=mock_episode_manager),
        ),
        patch(
            "bot.chat_pipeline.get_memory_service",
            return_value=mock_memory_service,
        ),
        patch(
            "bot.chat_pipeline.get_llm_service",
            return_value=mock_llm_service,
        ),
        patch(
            "bot.chat_pipeline.get_context_builder",
            return_value=mock_context_builder,
        ),
        patch(
            "bot.chat_pipeline.get_system_prompt",
            return_value="You are a helpful assistant.",
        ),
        patch(
            "bot.chat_pipeline.get_langfuse_service",
            return_value=mock_langfuse_service,
        ),
        patch(
            "bot.chat_pipeline.MEMORY_SERVICE_AVAILABLE",
            True,
        ),
        patch(
            "bot.handlers.DB_CLIENT_AVAILABLE",
            True,
        ),
        patch(
            "bot.chat_pipeline.DB_CLIENT_AVAILABLE",
            True,
        ),
        patch(
            "bot.handlers.get_db_client",
            return_value=mock_db_client,
        ),
        patch(
            "bot.chat_pipeline.get_db_client",
            return_value=mock_db_client,
        ),
    ):
        yield {
            "episode_manager": mock_episode_manager,
            "memory_service": mock_memory_service,
            "llm_service": mock_llm_service,
            "context_builder": mock_context_builder,
            "db_client": mock_db_client,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMemoryWriting:
    """Tests for memory writing in chat handler."""

    @pytest.fixture(autouse=True)
    def reset_memory_counters(self):
        """Reset memory write counters between tests."""
        from bot.chat_pipeline import _memory_write_counts

        _memory_write_counts.clear()
        yield
        _memory_write_counts.clear()

    @pytest.mark.asyncio
    async def test_chat_writes_memory_after_response(
        self, patched_handlers: dict[str, Any]
    ) -> None:
        """After LLM response, conversation is written to memory service via background task."""
        from bot.handlers import chat

        msg = MockMessage(text="I love cats", user_id=42)
        await chat(msg)

        # Let background tasks (asyncio.create_task) run
        await _drain_tasks()

        # Bot should have responded normally
        assert msg._last_answer == "test reply"

        # Memory write should have been called with content containing user + bot text
        patched_handlers["memory_service"].write_factual.assert_called_once()
        call_kwargs = patched_handlers["memory_service"].write_factual.call_args
        written_content = call_kwargs[1].get("content", call_kwargs[0][0] if call_kwargs[0] else "")
        assert "I love cats" in written_content
        assert "test reply" in written_content
        assert (
            call_kwargs[1].get("user_id", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
            == 42
        )

    @pytest.mark.asyncio
    async def test_chat_memory_write_error_swallowed(
        self, patched_handlers: dict[str, Any]
    ) -> None:
        """Memory write error is logged but doesn't break the response."""
        from bot.handlers import chat

        patched_handlers["memory_service"].write_factual = AsyncMock(
            side_effect=RuntimeError("memory write failed")
        )

        msg = MockMessage(text="hello there", user_id=42)
        await chat(msg)

        # Let background tasks run (the error is swallowed inside the task)
        await _drain_tasks()

        # Bot still responded despite memory write failure
        assert msg._last_answer == "test reply"

    @pytest.mark.asyncio
    async def test_chat_cognify_triggered_after_n_writes(
        self, patched_handlers: dict[str, Any]
    ) -> None:
        """After N writes, cognify() is triggered via asyncio.create_task."""
        from bot.handlers import chat

        with patch("bot.chat_pipeline._COGNIFY_EVERY_N_WRITES", 2):
            # First message — write_factual called, counter at 1 (no cognify)
            msg1 = MockMessage(text="first message", user_id=42)
            await chat(msg1)
            await _drain_tasks()
            patched_handlers["memory_service"].cognify.assert_not_called()

            # Second message — counter reaches 2, cognify triggered
            msg2 = MockMessage(text="second message", user_id=42)
            await chat(msg2)
            await _drain_tasks()
            patched_handlers["memory_service"].cognify.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_cognify_not_triggered_before_n(
        self, patched_handlers: dict[str, Any]
    ) -> None:
        """Before N writes, cognify is NOT triggered."""
        from bot.handlers import chat

        with patch("bot.chat_pipeline._COGNIFY_EVERY_N_WRITES", 10):
            msg = MockMessage(text="just one message", user_id=42)
            await chat(msg)
            await _drain_tasks()

            # Only 1 write, threshold is 10 — no cognify
            patched_handlers["memory_service"].cognify.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_no_memory_write_when_service_unavailable(
        self, patched_handlers: dict[str, Any]
    ) -> None:
        """When memory service is not available, no write attempt is made."""
        from bot.handlers import chat

        with patch("bot.chat_pipeline.MEMORY_SERVICE_AVAILABLE", False):
            msg = MockMessage(text="hello", user_id=42)
            await chat(msg)

            # Bot still responds normally
            assert msg._last_answer == "test reply"

            # write_factual should NOT have been called
            patched_handlers["memory_service"].write_factual.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_no_memory_write_on_llm_failure(
        self, patched_handlers: dict[str, Any]
    ) -> None:
        """When LLM fails (fallback response), no memory write."""
        from bot.handlers import chat

        patched_handlers["llm_service"].generate = AsyncMock(
            side_effect=RuntimeError("LLM exploded")
        )

        msg = MockMessage(text="hello", user_id=42)
        await chat(msg)

        # Bot should have responded with fallback
        assert msg._last_answer is not None
        assert "Прости" in msg._last_answer or "не получается" in msg._last_answer

        # No memory write when LLM failed (llm_response is None)
        patched_handlers["memory_service"].write_factual.assert_not_called()
