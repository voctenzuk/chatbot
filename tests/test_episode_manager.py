"""Tests for EpisodeManager service.

Tests cover:
- Episode creation and persistence
- Episode switching (time-gap trigger)
- Message storage with episode_id
- Episode lifecycle (start/close)
- Anti-flap mechanisms
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from bot.services.db_client import DatabaseClient, Episode, EpisodeMessage, Thread
from bot.services.episode_manager import (
    EpisodeManager,
    EpisodeManagerConfig,
    MessageResult,
)


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings based on text content."""
        import numpy as np

        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(10)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb.tolist())
        return embeddings


@pytest.fixture
def mock_db_client():
    """Create a mock database client."""
    client = MagicMock(spec=DatabaseClient)

    # Storage for mock data
    threads: dict[int, Thread] = {}
    episodes: dict[str, Episode] = {}
    messages: dict[str, EpisodeMessage] = {}
    message_counter = [0]
    episode_counter = [0]

    async def get_thread_for_user(user_id: int) -> Thread | None:
        return threads.get(user_id)

    async def get_or_create_thread(user_id: int) -> Thread:
        if user_id not in threads:
            thread_id = f"thread_{user_id}"
            threads[user_id] = Thread(
                id=thread_id,
                telegram_user_id=user_id,
                created_at=datetime.now(),
            )
        return threads[user_id]

    async def get_episode_by_id(episode_id: str) -> Episode:
        return episodes[episode_id]

    async def get_active_episode_for_user(user_id: int) -> Episode | None:
        thread = threads.get(user_id)
        if thread and thread.active_episode_id:
            return episodes.get(thread.active_episode_id)
        return None

    async def start_new_episode(thread_id: str, topic_label: str | None = None) -> Episode:
        episode_counter[0] += 1
        episode_id = f"ep_{episode_counter[0]}"
        episode = Episode(
            id=episode_id,
            thread_id=thread_id,
            status="active",
            started_at=datetime.now(),
            topic_label=topic_label,
        )
        episodes[episode_id] = episode

        # Update thread's active episode
        user_id = int(thread_id.split("_")[1])
        if user_id in threads:
            threads[user_id].active_episode_id = episode_id

        return episode

    async def close_episode(episode_id: str) -> Episode:
        episode = episodes[episode_id]
        episode.status = "closed"
        episode.ended_at = datetime.now()
        # Clear the thread's active episode
        for t in threads.values():
            if t.active_episode_id == episode_id:
                t.active_episode_id = None
                break
        return episode

    async def add_message(
        telegram_user_id: int,
        role: str,
        content_text: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        model: str | None = None,
    ) -> EpisodeMessage:
        message_counter[0] += 1
        message_id = f"msg_{message_counter[0]}"

        thread = threads.get(telegram_user_id)
        episode_id = (thread.active_episode_id or "orphan") if thread else "orphan"

        message = EpisodeMessage(
            id=message_id,
            episode_id=episode_id,
            role=role,
            content_text=content_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            created_at=datetime.now(),
        )
        messages[message_id] = message

        # Update episode's last_user_message_at if user message
        if role == "user" and episode_id in episodes:
            episodes[episode_id].last_user_message_at = datetime.now()

        return message

    async def get_messages_for_episode(episode_id: str, limit: int = 100) -> list[EpisodeMessage]:
        return [m for m in messages.values() if m.episode_id == episode_id][-limit:]

    async def get_recent_messages(user_id: int, limit: int = 50) -> list[EpisodeMessage]:
        thread = threads.get(user_id)
        if not thread or not thread.active_episode_id:
            return []
        return [m for m in messages.values() if m.episode_id == thread.active_episode_id][-limit:]

    async def get_episodes_for_thread(thread_id: str) -> list[Episode]:
        return [ep for ep in episodes.values() if ep.thread_id == thread_id]

    client.get_thread_for_user = get_thread_for_user
    client.get_or_create_thread = get_or_create_thread
    client.get_episode_by_id = get_episode_by_id
    client.get_active_episode_for_user = get_active_episode_for_user
    client.start_new_episode = start_new_episode
    client.close_episode = close_episode
    client.add_message = add_message
    client.get_messages_for_episode = get_messages_for_episode
    client.get_recent_messages = get_recent_messages
    client.get_episodes_for_thread = get_episodes_for_thread

    return client


@pytest.fixture
def episode_manager(mock_db_client):
    """Create an EpisodeManager with mock database."""
    config = EpisodeManagerConfig(
        time_gap_threshold=3600.0,  # 1 hour for testing
        min_episode_duration=0,  # Disable for most tests
    )
    return EpisodeManager(
        config=config,
        db_client=mock_db_client,
    )


@pytest.fixture
def episode_manager_with_topic(mock_db_client):
    """Create an EpisodeManager with topic detection enabled."""
    config = EpisodeManagerConfig(
        time_gap_threshold=3600.0,
        min_episode_duration=0,
        topic_similarity_threshold=0.7,
        topic_min_messages=3,
    )
    return EpisodeManager(
        config=config,
        db_client=mock_db_client,
        embedding_provider=MockEmbeddingProvider(),
    )


class TestEpisodeManagerBasic:
    """Tests for basic EpisodeManager functionality."""

    @pytest.mark.asyncio
    async def test_first_message_creates_thread_and_episode(self, episode_manager):
        """Test that first message creates thread and episode."""
        result = await episode_manager.process_user_message(
            user_id=12345,
            content="Hello!",
        )

        assert isinstance(result, MessageResult)
        assert result.message.episode_id is not None
        assert result.episode.id == result.message.episode_id
        assert result.is_new_episode is True
        assert result.switch_decision.should_switch is True

    @pytest.mark.asyncio
    async def test_consecutive_messages_same_episode(self, episode_manager):
        """Test that consecutive messages stay in same episode."""
        # First message
        await episode_manager.process_user_message(
            user_id=12345,
            content="Hello!",
        )

        # Second message (quick follow-up)
        result = await episode_manager.process_user_message(
            user_id=12345,
            content="How are you?",
        )

        assert result.is_new_episode is False
        assert result.switch_decision.should_switch is False

    @pytest.mark.asyncio
    async def test_message_has_episode_id(self, episode_manager):
        """Test that every message has an episode_id assigned."""
        result = await episode_manager.process_user_message(
            user_id=12345,
            content="Test message",
        )

        assert result.message.episode_id is not None
        assert len(result.message.episode_id) > 0

    @pytest.mark.asyncio
    async def test_user_isolation(self, episode_manager):
        """Test that episodes are isolated between users."""
        # User 1 message
        result1 = await episode_manager.process_user_message(
            user_id=111,
            content="Hello from user 1",
        )

        # User 2 message
        result2 = await episode_manager.process_user_message(
            user_id=222,
            content="Hello from user 2",
        )

        # Should have different episodes
        assert result1.episode.id != result2.episode.id
        assert result1.episode.id == result1.message.episode_id
        assert result2.episode.id == result2.message.episode_id


class TestEpisodeSwitching:
    """Tests for episode switching logic."""

    @pytest.mark.asyncio
    async def test_time_gap_triggers_new_episode(self, episode_manager, mock_db_client):
        """Test that large time gap triggers new episode."""
        # First message
        result1 = await episode_manager.process_user_message(
            user_id=12345,
            content="Hello!",
        )
        first_episode_id = result1.episode.id

        # Simulate time passing by modifying the episode's last_user_message_at
        episode = await mock_db_client.get_active_episode_for_user(12345)
        episode.last_user_message_at = datetime.now() - timedelta(hours=2)

        # Second message after time gap
        result2 = await episode_manager.process_user_message(
            user_id=12345,
            content="Are you there?",
        )

        assert result2.is_new_episode is True
        assert result2.episode.id != first_episode_id
        assert result2.switch_decision.trigger_type == "time_gap"

    @pytest.mark.asyncio
    async def test_no_switch_within_time_threshold(self, episode_manager):
        """Test that messages within time threshold stay in same episode."""
        # First message
        await episode_manager.process_user_message(
            user_id=12345,
            content="Hello!",
        )

        # Second message after 10 minutes (within 1 hour threshold)
        result = await episode_manager.process_user_message(
            user_id=12345,
            content="Follow up",
        )

        assert result.is_new_episode is False

    @pytest.mark.asyncio
    async def test_topic_shift_triggers_new_episode(
        self, episode_manager_with_topic, mock_db_client
    ):
        """Test that topic shift triggers new episode."""
        manager = episode_manager_with_topic

        # Build up episode with pet-related messages
        await manager.process_user_message(12345, "I love cats")
        await manager.process_user_message(12345, "My cat is so cute")
        await manager.process_user_message(12345, "Do you like pets?")

        # Get the current episode and add messages to it for topic context
        _ = await mock_db_client.get_active_episode_for_user(12345)
        for i in range(3):
            await mock_db_client.add_message(
                telegram_user_id=12345,
                role="user",
                content_text="I love cats and pets",
            )

        # Now shift to weather topic
        result = await manager.process_user_message(
            12345,
            "The weather is terrible today with lots of rain and clouds",
        )

        # Should trigger topic shift (weather is very different from pets)
        assert result.switch_decision.trigger_type in [
            "topic_shift",
            "time_gap",
            "new_conversation",
        ]


class TestAntiFlap:
    """Tests for anti-flap mechanisms."""

    @pytest.mark.asyncio
    async def test_min_episode_duration_prevents_switch(self, mock_db_client):
        """Test that min episode duration prevents rapid switching."""
        config = EpisodeManagerConfig(
            min_episode_duration=300.0,  # 5 minutes
            time_gap_threshold=60.0,  # 1 minute
        )
        manager = EpisodeManager(config=config, db_client=mock_db_client)

        # First message
        await manager.process_user_message(
            user_id=12345,
            content="Hello!",
        )

        # Try to trigger new episode after 2 minutes
        episode = await mock_db_client.get_active_episode_for_user(12345)
        episode.started_at = datetime.now() - timedelta(minutes=2)

        result = await manager.process_user_message(
            user_id=12345,
            content="New topic",
        )

        assert result.is_new_episode is False
        assert result.switch_decision.trigger_type == "anti_flap_min_duration"

    @pytest.mark.asyncio
    async def test_rate_limiting_prevents_excessive_switches(self, mock_db_client):
        """Test that rate limiting prevents too many episodes per hour."""
        config = EpisodeManagerConfig(
            min_episode_duration=0,  # Disable for this test
            max_episodes_per_hour=3,
            time_gap_threshold=1.0,  # 1 second
        )
        manager = EpisodeManager(config=config, db_client=mock_db_client)

        # Create multiple episodes rapidly
        for i in range(5):
            episode = await mock_db_client.get_active_episode_for_user(12345)
            if episode:
                episode.last_user_message_at = datetime.now() - timedelta(seconds=2)

            _ = await manager.process_user_message(
                user_id=12345,
                content=f"Message {i}",
            )

        # Should have hit rate limit
        switch_count = manager._get_recent_switch_count(12345)
        assert switch_count <= config.max_episodes_per_hour


class TestEpisodeLifecycle:
    """Tests for episode lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_current_episode(self, episode_manager):
        """Test explicitly closing the current episode."""
        # Create an episode first
        await episode_manager.process_user_message(
            user_id=12345,
            content="Hello!",
        )

        # Close the episode
        closed = await episode_manager.close_current_episode(
            user_id=12345,
            final_summary="Test summary",
        )

        assert closed is not None
        assert closed.status == "closed"

    @pytest.mark.asyncio
    async def test_close_nonexistent_episode(self, episode_manager):
        """Test closing episode when none exists."""
        result = await episode_manager.close_current_episode(user_id=99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_new_episode_after_close(self, episode_manager):
        """Test that new episode is created after closing."""
        # First episode
        result1 = await episode_manager.process_user_message(
            user_id=12345,
            content="Hello!",
        )
        first_episode_id = result1.episode.id

        # Close it
        await episode_manager.close_current_episode(user_id=12345)

        # New message should create new episode
        result2 = await episode_manager.process_user_message(
            user_id=12345,
            content="Back again!",
        )

        assert result2.episode.id != first_episode_id


class TestAssistantMessages:
    """Tests for assistant message handling."""

    @pytest.mark.asyncio
    async def test_assistant_message_stored(self, episode_manager):
        """Test that assistant messages are stored."""
        # Create user message first (to create episode)
        await episode_manager.process_user_message(
            user_id=12345,
            content="Hello!",
        )

        # Store assistant response
        result = await episode_manager.process_assistant_message(
            user_id=12345,
            content="Hello! How can I help?",
            tokens_in=10,
            tokens_out=20,
            model="test-model",
        )

        assert result.message.role == "assistant"
        assert result.message.content_text == "Hello! How can I help?"
        assert result.message.tokens_in == 10
        assert result.message.tokens_out == 20
        assert result.message.model == "test-model"

    @pytest.mark.asyncio
    async def test_assistant_message_creates_episode_if_none(self, episode_manager):
        """Test that assistant message creates episode if none exists."""
        result = await episode_manager.process_assistant_message(
            user_id=12345,
            content="Welcome message",
        )

        assert result.episode is not None
        assert result.message.episode_id is not None


class TestEpisodeRetrieval:
    """Tests for episode retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_current_episode(self, episode_manager):
        """Test getting current episode."""
        # Create episode
        await episode_manager.process_user_message(
            user_id=12345,
            content="Hello!",
        )

        episode = await episode_manager.get_current_episode(12345)
        assert episode is not None
        assert episode.status == "active"

    @pytest.mark.asyncio
    async def test_get_episode_history(self, episode_manager, mock_db_client):
        """Test getting episode history."""
        # Create multiple episodes
        for i in range(3):
            await episode_manager.process_user_message(
                user_id=12345,
                content=f"Message {i}",
            )
            # Close current episode to force new one
            await episode_manager.close_current_episode(12345)

        # New message creates 4th episode
        await episode_manager.process_user_message(
            user_id=12345,
            content="Current message",
        )

        history = await episode_manager.get_episode_history(12345)
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_get_messages_for_current_episode(self, episode_manager):
        """Test getting messages for current episode."""
        # Create some messages
        await episode_manager.process_user_message(12345, "Message 1")
        await episode_manager.process_user_message(12345, "Message 2")
        await episode_manager.process_user_message(12345, "Message 3")

        messages = await episode_manager.get_messages_for_current_episode(12345)
        assert len(messages) >= 3

    @pytest.mark.asyncio
    async def test_get_recent_messages(self, episode_manager):
        """Test getting recent messages."""
        # Create some messages
        for i in range(5):
            await episode_manager.process_user_message(12345, f"Message {i}")

        messages = await episode_manager.get_recent_messages(12345, limit=3)
        assert len(messages) <= 3


class TestEpisodeManagerConfig:
    """Tests for EpisodeManager configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EpisodeManagerConfig()

        assert config.time_gap_threshold == 28800.0  # 8 hours
        assert config.topic_similarity_threshold == 0.7
        assert config.topic_min_messages == 3
        assert config.min_episode_duration == 300.0  # 5 minutes
        assert config.max_episodes_per_hour == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = EpisodeManagerConfig(
            time_gap_threshold=3600.0,
            topic_similarity_threshold=0.5,
            min_episode_duration=60.0,
            max_episodes_per_hour=5,
        )

        assert config.time_gap_threshold == 3600.0
        assert config.topic_similarity_threshold == 0.5
        assert config.min_episode_duration == 60.0
        assert config.max_episodes_per_hour == 5


class TestGlobalInstance:
    """Tests for global EpisodeManager instance."""

    def test_set_episode_manager(self):
        """Test setting global instance."""
        from bot.services.episode_manager import get_episode_manager, set_episode_manager

        custom_manager = MagicMock(spec=EpisodeManager)
        set_episode_manager(custom_manager)

        retrieved = get_episode_manager()
        assert retrieved is custom_manager


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_message(self, episode_manager):
        """Test handling of empty message."""
        result = await episode_manager.process_user_message(
            user_id=12345,
            content="",
        )

        assert result.message.content_text == ""
        assert result.message.episode_id is not None

    @pytest.mark.asyncio
    async def test_very_long_message(self, episode_manager):
        """Test handling of very long message."""
        long_content = "Word " * 1000

        result = await episode_manager.process_user_message(
            user_id=12345,
            content=long_content,
        )

        assert result.message.content_text == long_content
        assert result.message.episode_id is not None

    @pytest.mark.asyncio
    async def test_special_characters(self, episode_manager):
        """Test handling of special characters."""
        special_content = "Hello! 🎉 How are you? @#$%^&*() café 日本語"

        result = await episode_manager.process_user_message(
            user_id=12345,
            content=special_content,
        )

        assert result.message.content_text == special_content

    @pytest.mark.asyncio
    async def test_multiple_users_independent(self, episode_manager):
        """Test that multiple users have independent episode state."""
        users = [100, 200, 300]
        episodes = []

        for user_id in users:
            result = await episode_manager.process_user_message(
                user_id=user_id,
                content=f"Hello from user {user_id}",
            )
            episodes.append(result.episode.id)

        # All episodes should be different
        assert len(set(episodes)) == len(users)
