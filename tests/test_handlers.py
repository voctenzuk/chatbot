"""Tests for bot handlers.

Tests cover:
- Message handling with episode persistence
- Episode switching integration in handlers
- Non-text message handling
- Graceful degradation when DB is unavailable
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from aiogram.types import Chat, Message, User

from bot.chat_pipeline import ChatPipeline, ChatResult
from bot.handlers import _extract_message_content, chat, start


class MockMessage:
    """Mock Telegram message for testing.

    This is intentionally loosely-typed because aiogram's Message model is complex.
    """

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
    ):
        self.text = text
        self.caption = caption
        self.from_user = MagicMock(spec=User)
        self.from_user.id = user_id
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
        """Mock answer method."""
        self._last_answer = text
        return MagicMock()


def _make_message_result() -> MagicMock:
    """Build a mock MessageResult."""
    mock_episode = MagicMock()
    mock_episode.id = str(uuid4())

    mock_msg = MagicMock()
    mock_msg.id = str(uuid4())
    mock_msg.episode_id = mock_episode.id

    mock_decision = MagicMock()
    mock_decision.should_switch = False
    mock_decision.reason = "Continuing current episode"
    mock_decision.confidence = 0.5
    mock_decision.trigger_type = None

    result = MagicMock()
    result.message = mock_msg
    result.episode = mock_episode
    result.is_new_episode = False
    result.switch_decision = mock_decision
    return result


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Create a mock ChatPipeline with necessary dependencies."""
    pipeline = MagicMock(spec=ChatPipeline)

    # Mock episode manager accessed through pipeline
    episode_manager = AsyncMock()
    episode_manager.process_user_message = AsyncMock(return_value=_make_message_result())
    episode_manager.process_assistant_message = AsyncMock(return_value=_make_message_result())
    pipeline._episode_manager = episode_manager
    pipeline._db_client = None

    # Mock handle_message for chat handler
    pipeline.handle_message = AsyncMock(
        return_value=ChatResult(response_text="test reply", image_bytes=None)
    )

    return pipeline


class TestStartHandler:
    """Tests for start command handler."""

    @pytest.mark.asyncio
    async def test_start_creates_episode(self, mock_pipeline: MagicMock) -> None:
        """Test that /start creates a new episode."""
        message = MockMessage(text="/start", user_id=12345)

        await start(message, pipeline=mock_pipeline)  # type: ignore[arg-type]

        # Should persist user message and assistant response
        assert mock_pipeline._episode_manager.process_user_message.call_count == 1
        assert mock_pipeline._episode_manager.process_assistant_message.call_count == 1

        # Check user message
        first_call = mock_pipeline._episode_manager.process_user_message.call_args
        assert first_call.kwargs["user_id"] == 12345
        assert first_call.kwargs["content"] == "/start"

    @pytest.mark.asyncio
    async def test_start_without_user(self, mock_pipeline: MagicMock) -> None:
        """Test start handler without from_user."""
        message = MockMessage(text="/start")
        message.from_user = None

        await start(message, pipeline=mock_pipeline)  # type: ignore[arg-type]

        # Should still work with user_id=0
        first_call = mock_pipeline._episode_manager.process_user_message.call_args
        assert first_call.kwargs["user_id"] == 0

    @pytest.mark.asyncio
    async def test_start_provisions_user_when_db_available(self, mock_pipeline: MagicMock) -> None:
        """Test that /start provisions user in DB when available."""
        mock_db = AsyncMock()
        mock_pipeline._db_client = mock_db

        message = MockMessage(text="/start", user_id=12345)
        await start(message, pipeline=mock_pipeline)  # type: ignore[arg-type]

        mock_db.provision_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_handles_db_unavailable(self, mock_pipeline: MagicMock) -> None:
        """Test that start handler informs user when DB is unavailable."""
        mock_pipeline._episode_manager.process_user_message = AsyncMock(
            side_effect=RuntimeError("Database client not configured")
        )

        message = MockMessage(text="/start", user_id=12345)
        await start(message, pipeline=mock_pipeline)  # type: ignore[arg-type]

        assert message._last_answer is not None
        assert "рядом" in message._last_answer
        assert "не сохраняются" in message._last_answer


class TestChatHandler:
    """Tests for chat message handler."""

    @pytest.mark.asyncio
    async def test_chat_delegates_to_pipeline(self, mock_pipeline: MagicMock) -> None:
        """Test that chat messages are delegated to the pipeline."""
        message = MockMessage(text="Hello bot!", user_id=12345)

        await chat(message, pipeline=mock_pipeline)  # type: ignore[arg-type]

        mock_pipeline.handle_message.assert_called_once_with(
            user_id=12345,
            content="Hello bot!",
            user_name=mock_pipeline.handle_message.call_args.kwargs["user_name"],
        )

    @pytest.mark.asyncio
    async def test_chat_returns_response(self, mock_pipeline: MagicMock) -> None:
        """Test that chat handler returns the pipeline response."""
        mock_pipeline.handle_message = AsyncMock(
            return_value=ChatResult(response_text="LLM reply text")
        )
        message = MockMessage(text="Hello!", user_id=12345)

        await chat(message, pipeline=mock_pipeline)  # type: ignore[arg-type]

        assert message._last_answer == "LLM reply text"

    @pytest.mark.asyncio
    async def test_chat_with_episode_switch(self, mock_pipeline: MagicMock) -> None:
        """Test chat handler with episode switching."""
        mock_pipeline.handle_message = AsyncMock(
            return_value=ChatResult(response_text="reply after switch")
        )

        message = MockMessage(text="New topic", user_id=12345)
        await chat(message, pipeline=mock_pipeline)  # type: ignore[arg-type]

        assert mock_pipeline.handle_message.called
        assert message._last_answer == "reply after switch"

    @pytest.mark.asyncio
    async def test_chat_runtime_error_returns_fallback(self, mock_pipeline: MagicMock) -> None:
        """Test that RuntimeError returns LLM fallback message."""
        mock_pipeline.handle_message = AsyncMock(
            side_effect=RuntimeError("Database client not configured")
        )

        message = MockMessage(text="Hello", user_id=12345)
        await chat(message, pipeline=mock_pipeline)  # type: ignore[arg-type]

        assert message._last_answer is not None
        assert "Прости" in message._last_answer or "не получается" in message._last_answer

    @pytest.mark.asyncio
    async def test_chat_unexpected_error_returns_generic_fallback(
        self, mock_pipeline: MagicMock
    ) -> None:
        """Test that unexpected errors return generic fallback."""
        mock_pipeline.handle_message = AsyncMock(side_effect=Exception("Unexpected error"))

        message = MockMessage(text="Hello", user_id=12345)
        await chat(message, pipeline=mock_pipeline)  # type: ignore[arg-type]

        assert message._last_answer is not None
        assert "услышала" in message._last_answer


class TestExtractMessageContent:
    """Tests for _extract_message_content function."""

    def test_text_message(self) -> None:
        """Test extracting content - text messages bypass this function."""
        message = MockMessage(text="")
        result = _extract_message_content(cast(Message, message))
        assert result == "[Non-text message]"

    def test_photo_with_caption(self) -> None:
        """Test extracting content from photo with caption."""
        message = MockMessage(caption="My vacation photo")
        message.photo = [MagicMock()]
        result = _extract_message_content(cast(Message, message))
        assert "[Caption: My vacation photo]" in result
        assert "[Photo attached]" in result

    def test_document(self) -> None:
        """Test extracting content from document."""
        message = MockMessage()
        message.document = MagicMock(file_name="report.pdf")
        result = _extract_message_content(cast(Message, message))
        assert "[Document: report.pdf]" in result

    def test_voice_message(self) -> None:
        """Test extracting content from voice message."""
        message = MockMessage()
        message.voice = MagicMock()
        result = _extract_message_content(cast(Message, message))
        assert "[Voice message]" in result

    def test_video(self) -> None:
        """Test extracting content from video."""
        message = MockMessage()
        message.video = MagicMock()
        result = _extract_message_content(cast(Message, message))
        assert "[Video attached]" in result

    def test_audio(self) -> None:
        """Test extracting content from audio."""
        message = MockMessage()
        message.audio = MagicMock()
        result = _extract_message_content(cast(Message, message))
        assert "[Audio attached]" in result

    def test_sticker(self) -> None:
        """Test extracting content from sticker."""
        message = MockMessage()
        message.sticker = MagicMock(emoji="😊")
        result = _extract_message_content(cast(Message, message))
        assert "[Sticker: 😊]" in result

    def test_location(self) -> None:
        """Test extracting content from location."""
        message = MockMessage()
        message.location = MagicMock(latitude=51.5074, longitude=-0.1278)
        result = _extract_message_content(cast(Message, message))
        assert "[Location: 51.5074, -0.1278]" in result

    def test_contact(self) -> None:
        """Test extracting content from contact."""
        message = MockMessage()
        message.contact = MagicMock(first_name="John")
        result = _extract_message_content(cast(Message, message))
        assert "[Contact: John]" in result

    def test_sticker_without_emoji(self) -> None:
        """Test extracting content from sticker without emoji."""
        message = MockMessage()
        message.sticker = MagicMock(emoji=None)
        result = _extract_message_content(cast(Message, message))
        assert "[Sticker: emoji]" in result

    def test_contact_without_name(self) -> None:
        """Test extracting content from contact without name."""
        message = MockMessage()
        message.contact = MagicMock(first_name=None)
        result = _extract_message_content(cast(Message, message))
        assert "[Contact: unnamed]" in result

    def test_empty_message(self) -> None:
        """Test extracting content from empty message."""
        message = MockMessage()
        result = _extract_message_content(cast(Message, message))
        assert result == "[Non-text message]"

    def test_multiple_attachments(self) -> None:
        """Test extracting content with multiple attachments."""
        message = MockMessage(caption="Check this out")
        message.photo = [MagicMock()]
        message.document = MagicMock(file_name="file.txt")
        result = _extract_message_content(cast(Message, message))
        assert "[Caption: Check this out]" in result
        assert "[Photo attached]" in result
        assert "[Document: file.txt]" in result
