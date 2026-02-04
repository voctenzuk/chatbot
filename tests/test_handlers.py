"""Tests for bot handlers.

Tests cover:
- Message handling with episode persistence
- Episode switching integration in handlers
- Non-text message handling
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiogram.types import Chat, Message, User

# Import handlers module
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

    async def answer(self, text: str, **kwargs):
        """Mock answer method."""
        self._last_answer = text
        return MagicMock()


@pytest.fixture
def mock_episode_manager():
    """Create a mock episode manager."""
    manager = AsyncMock()

    # Mock Episode
    mock_episode = MagicMock()
    mock_episode.id = str(uuid4())

    # Mock EpisodeMessage
    mock_message = MagicMock()
    mock_message.id = str(uuid4())
    mock_message.episode_id = mock_episode.id

    # Mock SwitchDecision
    mock_decision = MagicMock()
    mock_decision.should_switch = False
    mock_decision.reason = "Continuing current episode"
    mock_decision.confidence = 0.5
    mock_decision.trigger_type = None

    # Create MessageResult-like return
    mock_result = MagicMock()
    mock_result.message = mock_message
    mock_result.episode = mock_episode
    mock_result.is_new_episode = False
    mock_result.switch_decision = mock_decision

    manager.process_user_message = AsyncMock(return_value=mock_result)
    manager.process_assistant_message = AsyncMock(return_value=mock_result)

    return manager


@pytest.fixture
def mock_get_episode_manager(mock_episode_manager):
    """Mock get_episode_manager_service function."""
    with patch("bot.handlers.get_episode_manager_service", return_value=mock_episode_manager):
        yield mock_episode_manager


class TestStartHandler:
    """Tests for start command handler."""

    @pytest.mark.asyncio
    async def test_start_creates_episode(self, mock_get_episode_manager):
        """Test that /start creates a new episode."""
        message = MockMessage(text="/start", user_id=12345)

        await start(message)

        # Should persist user message and assistant response
        assert mock_get_episode_manager.process_user_message.call_count == 1
        assert mock_get_episode_manager.process_assistant_message.call_count == 1

        # Check user message
        first_call = mock_get_episode_manager.process_user_message.call_args
        assert first_call.kwargs["user_id"] == 12345
        assert first_call.kwargs["content"] == "/start"

    @pytest.mark.asyncio
    async def test_start_without_user(self, mock_get_episode_manager):
        """Test start handler without from_user."""
        message = MockMessage(text="/start")
        message.from_user = None

        await start(message)

        # Should still work with user_id=0
        first_call = mock_get_episode_manager.process_user_message.call_args
        assert first_call.kwargs["user_id"] == 0


class TestChatHandler:
    """Tests for chat message handler."""

    @pytest.mark.asyncio
    async def test_chat_persists_message(self, mock_get_episode_manager):
        """Test that chat messages are persisted."""
        message = MockMessage(text="Hello bot!", user_id=12345)

        await chat(message)

        # Should persist user message
        mock_get_episode_manager.process_user_message.assert_called_once()
        call_args = mock_get_episode_manager.process_user_message.call_args
        assert call_args.kwargs["user_id"] == 12345
        assert call_args.kwargs["content"] == "Hello bot!"

        # Should persist assistant response
        mock_get_episode_manager.process_assistant_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_episode_switch(self, mock_get_episode_manager):
        """Test chat handler with episode switching."""
        # Create a mock result with is_new_episode=True
        mock_result = MagicMock()
        mock_result.message.episode_id = str(uuid4())
        mock_result.episode.id = str(uuid4())
        mock_result.is_new_episode = True
        mock_result.switch_decision.should_switch = True
        mock_result.switch_decision.reason = "Time gap detected"
        mock_result.switch_decision.trigger_type = "time_gap"

        mock_get_episode_manager.process_user_message = AsyncMock(return_value=mock_result)

        message = MockMessage(text="New topic", user_id=12345)
        await chat(message)

        # Should still work with episode switch
        assert mock_get_episode_manager.process_user_message.called


class TestExtractMessageContent:
    """Tests for _extract_message_content function."""

    def test_text_message(self):
        """Test extracting content - text messages bypass this function."""
        # Note: _extract_message_content is only called when message.text is empty
        # A text message with content would be handled directly by the handler
        message = MockMessage(text="")  # Empty text triggers content extraction
        result = _extract_message_content(cast(Message, message))
        assert result == "[Non-text message]"

    def test_photo_with_caption(self):
        """Test extracting content from photo with caption."""
        message = MockMessage(caption="My vacation photo")
        message.photo = [MagicMock()]  # Non-empty photo array
        result = _extract_message_content(cast(Message, message))
        assert "[Caption: My vacation photo]" in result
        assert "[Photo attached]" in result

    def test_document(self):
        """Test extracting content from document."""
        message = MockMessage()
        message.document = MagicMock(file_name="report.pdf")
        result = _extract_message_content(cast(Message, message))
        assert "[Document: report.pdf]" in result

    def test_voice_message(self):
        """Test extracting content from voice message."""
        message = MockMessage()
        message.voice = MagicMock()
        result = _extract_message_content(cast(Message, message))
        assert "[Voice message]" in result

    def test_video(self):
        """Test extracting content from video."""
        message = MockMessage()
        message.video = MagicMock()
        result = _extract_message_content(cast(Message, message))
        assert "[Video attached]" in result

    def test_audio(self):
        """Test extracting content from audio."""
        message = MockMessage()
        message.audio = MagicMock()
        result = _extract_message_content(cast(Message, message))
        assert "[Audio attached]" in result

    def test_sticker(self):
        """Test extracting content from sticker."""
        message = MockMessage()
        message.sticker = MagicMock(emoji="😊")
        result = _extract_message_content(cast(Message, message))
        assert "[Sticker: 😊]" in result

    def test_location(self):
        """Test extracting content from location."""
        message = MockMessage()
        message.location = MagicMock(latitude=51.5074, longitude=-0.1278)
        result = _extract_message_content(cast(Message, message))
        assert "[Location: 51.5074, -0.1278]" in result

    def test_contact(self):
        """Test extracting content from contact."""
        message = MockMessage()
        message.contact = MagicMock(first_name="John")
        result = _extract_message_content(cast(Message, message))
        assert "[Contact: John]" in result

    def test_sticker_without_emoji(self):
        """Test extracting content from sticker without emoji."""
        message = MockMessage()
        message.sticker = MagicMock(emoji=None)
        result = _extract_message_content(cast(Message, message))
        assert "[Sticker: emoji]" in result

    def test_contact_without_name(self):
        """Test extracting content from contact without name."""
        message = MockMessage()
        message.contact = MagicMock(first_name=None)
        result = _extract_message_content(cast(Message, message))
        assert "[Contact: unnamed]" in result

    def test_empty_message(self):
        """Test extracting content from empty message."""
        message = MockMessage()
        result = _extract_message_content(cast(Message, message))
        assert result == "[Non-text message]"

    def test_multiple_attachments(self):
        """Test extracting content with multiple attachments."""
        message = MockMessage(caption="Check this out")
        message.photo = [MagicMock()]
        message.document = MagicMock(file_name="file.txt")
        result = _extract_message_content(cast(Message, message))
        assert "[Caption: Check this out]" in result
        assert "[Photo attached]" in result
        assert "[Document: file.txt]" in result
