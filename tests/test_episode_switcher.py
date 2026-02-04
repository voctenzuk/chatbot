"""Tests for Episode Switcher module.

Tests cover:
- Time-gap trigger detection
- Topic shift detection using embedding similarity
- Anti-flap mechanisms (rate limiting, minimum duration)
- Synthetic dialogue scenarios
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from bot.services.episode_switcher import (
    EpisodeConfig,
    EpisodeManager,
    SimpleEmbeddingProvider,
    OpenAIEmbeddingProvider,
    Message,
    Episode,
    SwitchDecision,
    get_episode_manager,
    set_episode_manager,
)


class TestSimpleEmbeddingProvider:
    """Tests for SimpleEmbeddingProvider (TF-IDF based)."""

    @pytest.fixture
    def provider(self):
        """Create fresh embedding provider."""
        return SimpleEmbeddingProvider()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_embed_single_text(self, provider):
        """Test embedding a single text."""
        embeddings = await provider.embed(["Hello world"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0
        # Should be normalized
        assert abs(np.linalg.norm(embeddings[0]) - 1.0) < 1e-6 or np.linalg.norm(embeddings[0]) == 0

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, provider):
        """Test embedding multiple texts."""
        # Use more distinct texts to ensure clear differentiation
        texts = ["Hello world today", "Goodbye world tomorrow", "Hello world today"]
        embeddings = await provider.embed(texts)
    
        assert len(embeddings) == 3
        # Identical texts should have perfect similarity
        sim_identical = np.dot(embeddings[0], embeddings[2])  # Same text
        assert sim_identical > 0.99  # Should be nearly identical

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, provider):
        """Test embedding empty list."""
        embeddings = await provider.embed([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_similarity_semantic_topics(self, provider):
        """Test that similar topics have higher similarity."""
        # Use more distinct texts to ensure TF-IDF can differentiate
        texts = [
            "I love cats and dogs pets animals fur",  # pets
            "My cat and dog are sleeping together pets",  # pets (similar)
            "The weather forecast rain sunshine clouds sky temperature",  # weather (different)
        ]
        embeddings = await provider.embed(texts)

        # Calculate similarities
        def cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_pets = cos_sim(embeddings[0], embeddings[1])  # pet texts
        sim_diff = cos_sim(embeddings[0], embeddings[2])  # pet vs weather

        # Similar topics should have some similarity, different topics less
        # With more distinct vocabularies, this should hold
        assert sim_pets >= sim_diff, f"Pet texts should be more similar: {sim_pets} vs {sim_diff}"


class TestEpisodeConfig:
    """Tests for EpisodeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EpisodeConfig()

        assert config.time_gap_threshold == 3600.0  # 1 hour
        assert config.topic_similarity_threshold == 0.7
        assert config.topic_min_messages == 3
        assert config.min_episode_duration == 300.0  # 5 minutes
        assert config.max_episodes_per_hour == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = EpisodeConfig(
            time_gap_threshold=1800.0,
            topic_similarity_threshold=0.5,
            min_episode_duration=60.0,
        )

        assert config.time_gap_threshold == 1800.0
        assert config.topic_similarity_threshold == 0.5
        assert config.min_episode_duration == 60.0


class TestEpisodeManagerBasic:
    """Tests for EpisodeManager basic functionality."""

    @pytest.fixture
    def manager(self):
        """Create fresh episode manager."""
        return EpisodeManager()

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create mock embedding provider with controllable embeddings."""
        provider = MagicMock()

        # Return deterministic embeddings based on text content hash
        async def mock_embed(texts):
            embeddings = []
            for text in texts:
                # Create a simple deterministic embedding
                np.random.seed(hash(text) % 2**32)
                emb = np.random.randn(10)
                emb = emb / np.linalg.norm(emb)  # normalize
                embeddings.append(emb.tolist())
            return embeddings

        provider.embed = mock_embed
        return provider

    @pytest.mark.asyncio
    async def test_first_message_starts_episode(self, manager):
        """Test that first message starts a new episode."""
        message, decision = await manager.add_message(
            user_id=1,
            content="Hello!",
            timestamp=datetime.now(),
        )

        assert decision.should_switch is True
        assert decision.trigger_type == "new_conversation"
        assert message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_consecutive_messages_same_episode(self, manager):
        """Test that consecutive messages stay in same episode."""
        now = datetime.now()

        # First message
        await manager.add_message(user_id=1, content="Hello!", timestamp=now)

        # Second message (quick follow-up)
        msg2, decision = await manager.add_message(
            user_id=1,
            content="How are you?",
            timestamp=now + timedelta(seconds=10),
        )

        assert decision.should_switch is False
        # When episode is too young, anti-flap prevents switching
        assert decision.trigger_type == "anti_flap_min_duration"

    @pytest.mark.asyncio
    async def test_time_gap_trigger(self, manager):
        """Test that large time gap triggers new episode."""
        now = datetime.now()

        # First message
        await manager.add_message(user_id=1, content="Hello!", timestamp=now)

        # Wait for minimum episode duration, then time gap
        future = now + timedelta(seconds=400) + timedelta(hours=2)  # > 5min + > 1hour

        msg, decision = await manager.add_message(
            user_id=1,
            content="Are you there?",
            timestamp=future,
        )

        assert decision.should_switch is True
        assert decision.trigger_type == "time_gap"

    @pytest.mark.asyncio
    async def test_anti_flap_min_duration(self, manager):
        """Test anti-flap minimum episode duration protection."""
        now = datetime.now()

        # First message
        await manager.add_message(user_id=1, content="Hello!", timestamp=now)

        # Try to trigger new episode after only 1 minute (too soon)
        future = now + timedelta(minutes=1)

        msg, decision = await manager.add_message(
            user_id=1,
            content="New topic completely different",
            timestamp=future,
        )

        assert decision.should_switch is False
        assert decision.trigger_type == "anti_flap_min_duration"

    @pytest.mark.asyncio
    async def test_anti_flap_rate_limiting(self, manager):
        """Test anti-flap rate limiting protection."""
        now = datetime.now()

        # Create many episodes quickly to trigger rate limit
        for i in range(15):
            ts = now + timedelta(minutes=i * 10)  # Every 10 minutes
            msg, decision = await manager.add_message(
                user_id=1,
                content=f"Message {i}",
                timestamp=ts,
            )

        # After 10 episodes, should hit rate limit
        # The last few should have been blocked by anti-flap
        switch_count = manager._get_recent_switch_count(1)
        assert switch_count <= manager.config.max_episodes_per_hour

    @pytest.mark.asyncio
    async def test_get_current_episode(self, manager):
        """Test retrieving current episode."""
        now = datetime.now()

        await manager.add_message(user_id=1, content="Hello!", timestamp=now)

        episode = manager.get_current_episode(1)

        assert episode is not None
        assert episode.user_id == 1
        assert episode.message_count == 1

    @pytest.mark.asyncio
    async def test_get_all_episodes(self, manager):
        """Test retrieving all episodes."""
        now = datetime.now()

        # Create some messages in first episode
        for i in range(3):
            await manager.add_message(
                user_id=1,
                content=f"Msg {i}",
                timestamp=now + timedelta(minutes=i * 6),  # > 5min apart
            )

        episodes = manager.get_all_episodes(1)
        assert len(episodes) >= 1

    @pytest.mark.asyncio
    async def test_clear_user_episodes(self, manager):
        """Test clearing user episodes."""
        now = datetime.now()

        await manager.add_message(user_id=1, content="Hello!", timestamp=now)
        manager.clear_user_episodes(1)

        assert manager.get_current_episode(1) is None
        assert manager.get_episodes(1) == []


class TestTopicShiftDetection:
    """Tests for topic shift detection using embeddings."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider with controlled embeddings."""
        provider = MagicMock()

        # Simulate embeddings: similar texts = similar vectors
        async def mock_embed(texts):
            embeddings = []
            for text in texts:
                text_lower = text.lower()
                if "cat" in text_lower or "pet" in text_lower or "kitten" in text_lower:
                    # Pet topic cluster
                    emb = [0.9, 0.1, 0.0, 0.0, 0.0]
                elif "weather" in text_lower or "rain" in text_lower or "sunny" in text_lower:
                    # Weather topic cluster
                    emb = [0.0, 0.9, 0.1, 0.0, 0.0]
                elif "work" in text_lower or "job" in text_lower or "office" in text_lower:
                    # Work topic cluster
                    emb = [0.0, 0.0, 0.9, 0.1, 0.0]
                else:
                    # Default/neutral
                    emb = [0.2, 0.2, 0.2, 0.2, 0.2]

                # Normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = [e / norm for e in emb]
                embeddings.append(emb)
            return embeddings

        provider.embed = mock_embed
        return provider

    @pytest.fixture
    def manager_with_mock(self, mock_provider):
        """Create manager with mock embedding provider."""
        config = EpisodeConfig(
            topic_similarity_threshold=0.7,
            topic_min_messages=3,
            min_episode_duration=0,  # Disable for testing
        )
        return EpisodeManager(config=config, embedding_provider=mock_provider)

    @pytest.mark.asyncio
    async def test_topic_shift_triggers_new_episode(self, manager_with_mock):
        """Test that topic shift triggers new episode."""
        now = datetime.now()

        # Build up episode with pet-related messages
        await manager_with_mock.add_message(user_id=1, content="I love my cat", timestamp=now)
        await manager_with_mock.add_message(
            user_id=1, content="My cat is so cute", timestamp=now + timedelta(minutes=1)
        )
        await manager_with_mock.add_message(
            user_id=1, content="Do you like pets?", timestamp=now + timedelta(minutes=2)
        )

        # Now shift to weather topic
        msg, decision = await manager_with_mock.add_message(
            user_id=1,
            content="The weather is nice today",
            timestamp=now + timedelta(minutes=3),
        )

        assert decision.should_switch is True
        assert decision.trigger_type == "topic_shift"

    @pytest.mark.asyncio
    async def test_similar_topic_no_switch(self, manager_with_mock):
        """Test that similar topic continues same episode."""
        now = datetime.now()

        # Build up episode with pet-related messages
        await manager_with_mock.add_message(user_id=1, content="I love my cat", timestamp=now)
        await manager_with_mock.add_message(
            user_id=1, content="My cat is so cute", timestamp=now + timedelta(minutes=1)
        )
        await manager_with_mock.add_message(
            user_id=1, content="Do you like pets?", timestamp=now + timedelta(minutes=2)
        )

        # Similar topic (another pet message)
        msg, decision = await manager_with_mock.add_message(
            user_id=1,
            content="I also have a kitten",
            timestamp=now + timedelta(minutes=3),
        )

        assert decision.should_switch is False

    @pytest.mark.asyncio
    async def test_not_enough_messages_no_topic_switch(self, manager_with_mock):
        """Test that topic shift doesn't trigger before min_messages."""
        now = datetime.now()

        # Only 2 messages
        await manager_with_mock.add_message(user_id=1, content="I love my cat", timestamp=now)
        await manager_with_mock.add_message(
            user_id=1, content="Weather is nice", timestamp=now + timedelta(minutes=1)
        )

        # Topic shift shouldn't trigger with only 2 messages
        msg, decision = await manager_with_mock.add_message(
            user_id=1,
            content="Work is stressful",
            timestamp=now + timedelta(minutes=2),
        )

        # Should not be topic_shift due to min_messages
        if decision.trigger_type == "topic_shift":
            pytest.fail("Topic shift shouldn't trigger with less than min_messages")


class TestSyntheticDialogues:
    """Tests using synthetic dialogues simulating real conversations."""

    @pytest.fixture
    def manager(self):
        """Create episode manager."""
        return EpisodeManager()

    @pytest.mark.asyncio
    async def test_casual_conversation_flow(self, manager):
        """Test a casual conversation without major topic shifts."""
        base_time = datetime.now()

        dialogue = [
            ("Hello! How are you?", 10),
            ("I'm doing great, thanks!", 15),
            ("What have you been up to?", 45),
            ("Just working on some projects", 60),
            ("That sounds interesting", 90),
            ("Yeah, it's been busy lately", 120),
        ]

        switch_count = 0
        for i, (content, delay_sec) in enumerate(dialogue):
            ts = base_time + timedelta(seconds=delay_sec)
            msg, decision = await manager.add_message(
                user_id=1, content=content, timestamp=ts
            )
            if decision.should_switch:
                switch_count += 1

        # Should mostly stay in one episode for casual chat
        assert switch_count <= 2  # May have initial switch

    @pytest.mark.asyncio
    async def test_topic_change_conversation(self, manager):
        """Test conversation with clear topic changes."""
        base_time = datetime.now()

        # Topics: greetings -> work -> hobbies -> weather
        dialogue = [
            ("Hey there!", 0),
            ("Hi! How's it going?", 10),
            ("Good! Let me tell you about my job...", 20),
            ("I work in software engineering", 30),
            ("The codebase is challenging", 40),
            ("By the way, I love playing guitar", 360),  # 6 min gap + topic change
            ("Music is my passion", 380),
            ("I practice every day", 400),
            ("Speaking of which, is it raining outside?", 720),  # 5+ min gap + topic change
            ("The weather forecast said it might", 740),
        ]

        decisions = []
        for content, delay_sec in dialogue:
            ts = base_time + timedelta(seconds=delay_sec + 300)  # Add 5 min to each for min_duration
            msg, decision = await manager.add_message(
                user_id=1, content=content, timestamp=ts
            )
            decisions.append(decision)

        # Count actual switches
        switches = [d for d in decisions if d.should_switch]

        # Should have switches at major topic changes + time gaps
        assert len(switches) >= 2

    @pytest.mark.asyncio
    async def test_interrupted_conversation(self, manager):
        """Test conversation with long interruptions."""
        base_time = datetime.now()

        # Morning conversation
        await manager.add_message(
            user_id=1, content="Good morning!", timestamp=base_time
        )
        await manager.add_message(
            user_id=1, content="How did you sleep?", timestamp=base_time + timedelta(minutes=6)
        )

        # Long break (work day)
        evening = base_time + timedelta(hours=9)
        msg, decision = await manager.add_message(
            user_id=1, content="Hey, I'm back from work!", timestamp=evening
        )

        # Should start new episode after long gap
        assert decision.should_switch is True
        assert decision.trigger_type == "time_gap"

    @pytest.mark.asyncio
    async def test_rapid_fire_messages(self, manager):
        """Test rapid-fire messages in short succession."""
        base_time = datetime.now()

        # Many messages in quick succession
        for i in range(20):
            msg, decision = await manager.add_message(
                user_id=1,
                content=f"Message {i}",
                timestamp=base_time + timedelta(seconds=i * 5),  # Every 5 seconds
            )
            # Should not switch episodes
            if i > 0:
                assert decision.should_switch is False

    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, manager):
        """Test that episodes are isolated between users."""
        base_time = datetime.now()

        # User 1 messages
        await manager.add_message(user_id=1, content="Hello from user 1", timestamp=base_time)

        # User 2 messages (interleaved)
        await manager.add_message(
            user_id=2, content="Hello from user 2", timestamp=base_time + timedelta(seconds=10)
        )

        # More user 1 messages
        await manager.add_message(
            user_id=1, content="Continuing my conversation", timestamp=base_time + timedelta(seconds=20)
        )

        # Check isolation
        user1_episodes = manager.get_all_episodes(1)
        user2_episodes = manager.get_all_episodes(2)

        # User 1 should have 1 episode with 2 messages
        assert len(user1_episodes) >= 1
        assert user1_episodes[-1].message_count == 2

        # User 2 should have 1 episode with 1 message
        assert len(user2_episodes) >= 1
        assert user2_episodes[-1].message_count == 1


class TestEpisodeSummary:
    """Tests for episode summary generation."""

    @pytest.fixture
    def manager(self):
        return EpisodeManager()

    @pytest.mark.asyncio
    async def test_empty_episode_summary(self, manager):
        """Test summary of empty episode."""
        episode = Episode(
            episode_id="test_1",
            user_id=1,
            start_time=datetime.now(),
            end_time=datetime.now(),
            messages=[],
        )

        summary = await manager.get_episode_summary(episode)
        assert "Empty episode" in summary

    @pytest.mark.asyncio
    async def test_episode_with_messages_summary(self, manager):
        """Test summary of episode with messages."""
        now = datetime.now()

        episode = Episode(
            episode_id="test_1",
            user_id=1,
            start_time=now,
            end_time=now + timedelta(minutes=10),
            messages=[
                Message(content="Hello there", timestamp=now, user_id=1),
                Message(content="How are you doing today", timestamp=now + timedelta(minutes=5), user_id=1),
            ],
        )

        summary = await manager.get_episode_summary(episode)
        assert "2 messages" in summary
        assert "10.0min" in summary or "10.min" in summary


class TestGlobalEpisodeManager:
    """Tests for global episode manager instance."""

    def test_get_episode_manager_creates_instance(self):
        """Test that get_episode_manager creates default instance."""
        set_episode_manager(None)  # Reset
        manager = get_episode_manager()

        assert isinstance(manager, EpisodeManager)

    def test_set_episode_manager(self):
        """Test setting global instance."""
        custom_manager = EpisodeManager()
        set_episode_manager(custom_manager)

        retrieved = get_episode_manager()
        assert retrieved is custom_manager


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def manager(self):
        return EpisodeManager()

    @pytest.mark.asyncio
    async def test_very_long_message(self, manager):
        """Test handling of very long messages."""
        long_content = "Word " * 10000

        msg, decision = await manager.add_message(
            user_id=1, content=long_content, timestamp=datetime.now()
        )

        assert msg.content == long_content
        assert decision.should_switch is True  # First message

    @pytest.mark.asyncio
    async def test_special_characters(self, manager):
        """Test handling of special characters."""
        special_content = "Hello! 🎉 How are you? @#$%^&*() café 日本語"

        msg, decision = await manager.add_message(
            user_id=1, content=special_content, timestamp=datetime.now()
        )

        assert msg.content == special_content

    @pytest.mark.asyncio
    async def test_empty_message(self, manager):
        """Test handling of empty message."""
        msg, decision = await manager.add_message(
            user_id=1, content="", timestamp=datetime.now()
        )

        assert msg.content == ""

    @pytest.mark.asyncio
    async def test_multiple_switches_with_clearing(self, manager):
        """Test multiple switches with clearing in between."""
        base_time = datetime.now()

        # Create some episodes
        for i in range(5):
            ts = base_time + timedelta(hours=i)
            await manager.add_message(user_id=1, content=f"Batch 1 - {i}", timestamp=ts)

        # Clear
        manager.clear_user_episodes(1)

        # Create more episodes
        for i in range(3):
            ts = base_time + timedelta(hours=i)
            await manager.add_message(user_id=1, content=f"Batch 2 - {i}", timestamp=ts)

        episodes = manager.get_all_episodes(1)
        # Should only have Batch 2 episodes
        assert all("Batch 2" in ep.messages[0].content for ep in episodes if ep.messages)

    @pytest.mark.asyncio
    async def test_simultaneous_topic_and_time_trigger(self, manager):
        """Test when both topic shift and time gap occur."""
        base_time = datetime.now()

        # First establish a topic
        await manager.add_message(
            user_id=1, content="I love discussing cats and pets",
            timestamp=base_time
        )
        await manager.add_message(
            user_id=1, content="My cat is adorable",
            timestamp=base_time + timedelta(minutes=6)
        )
        await manager.add_message(
            user_id=1, content="Do you have any pets?",
            timestamp=base_time + timedelta(minutes=12)
        )

        # Long time gap + topic shift
        future = base_time + timedelta(hours=3)
        msg, decision = await manager.add_message(
            user_id=1, content="The weather is terrible today",
            timestamp=future
        )

        # Should trigger due to time gap at minimum
        assert decision.should_switch is True
        assert decision.trigger_type in ["time_gap", "combined"]
