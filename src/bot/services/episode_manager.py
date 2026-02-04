"""EpisodeManager service with database persistence.

This module provides episode management with:
- Automatic episode switching based on time gap and topic shift
- Database persistence for episodes and messages
- Integration with Supabase for thread/episode/message storage
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from bot.services.episode_switcher import (
    EpisodeConfig,
    EpisodeManager as InMemoryEpisodeManager,
    SwitchDecision,
)

from bot.services.db_client import DatabaseClient, Episode, EpisodeMessage

DB_CLIENT_AVAILABLE = True


@dataclass
class EpisodeManagerConfig:
    """Configuration for EpisodeManager."""

    time_gap_threshold: float = 28800.0  # 8 hours default
    topic_similarity_threshold: float = 0.7
    topic_min_messages: int = 3
    min_episode_duration: float = 300.0  # 5 minutes
    max_episodes_per_hour: int = 10

    def to_episode_config(self) -> EpisodeConfig:
        """Convert to EpisodeConfig for episode_switcher."""
        return EpisodeConfig(
            time_gap_threshold=self.time_gap_threshold,
            topic_similarity_threshold=self.topic_similarity_threshold,
            topic_min_messages=self.topic_min_messages,
            min_episode_duration=self.min_episode_duration,
            max_episodes_per_hour=self.max_episodes_per_hour,
        )


@dataclass
class MessageResult:
    """Result of processing a message."""

    message: "EpisodeMessage"
    episode: "Episode"
    is_new_episode: bool
    switch_decision: SwitchDecision


class EpisodeManager:
    """Episode manager with database persistence.

    This service combines the in-memory episode switching logic with
    database persistence for threads, episodes, and messages.
    """

    def __init__(
        self,
        config: EpisodeManagerConfig | None = None,
        db_client: "DatabaseClient | None" = None,
        embedding_provider=None,
    ) -> None:
        self.config = config or EpisodeManagerConfig()
        self.db = db_client
        self._embedding_provider = embedding_provider
        self._in_memory_manager = InMemoryEpisodeManager(
            config=self.config.to_episode_config(),
            embedding_provider=embedding_provider,
        )
        # Track switch history for rate limiting
        self._switch_history: dict[int, list[datetime]] = {}

    def _get_recent_switch_count(self, user_id: int, window_hours: float = 1.0) -> int:
        """Count recent episode switches for anti-flap."""
        from datetime import timedelta

        if user_id not in self._switch_history:
            return 0

        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [t for t in self._switch_history[user_id] if t > cutoff]
        self._switch_history[user_id] = recent
        return len(recent)

    def _record_switch(self, user_id: int) -> None:
        """Record an episode switch for rate limiting."""
        if user_id not in self._switch_history:
            self._switch_history[user_id] = []
        self._switch_history[user_id].append(datetime.now())

    async def _seed_in_memory_state_from_db(self, user_id: int) -> bool:
        """Seed in-memory episode manager from database state.

        This prevents unnecessary episode splits after restart when
        an active episode exists in DB but in-memory state is empty.

        Args:
            user_id: Telegram user ID

        Returns:
            True if state was seeded successfully, False otherwise.
        """
        if self.db is None:
            return False

        episode = await self.db.get_active_episode_for_user(user_id)
        if episode is None:
            return False

        # Get messages for this episode to seed the in-memory state
        messages = await self.db.get_messages_for_episode(episode.id, limit=50)

        # Import here to avoid circular imports
        from bot.services.episode_switcher import Episode as InMemoryEpisode, Message

        # Create in-memory episode
        in_memory_episode = InMemoryEpisode(
            episode_id=episode.id,
            user_id=user_id,
            start_time=episode.started_at or datetime.now(),
            end_time=episode.last_user_message_at or episode.started_at or datetime.now(),
            messages=[
                Message(
                    content=m.content_text,
                    timestamp=m.created_at or datetime.now(),
                    user_id=user_id,
                    role=m.role,
                )
                for m in messages
            ],
        )

        # Seed the in-memory manager
        self._in_memory_manager._current_episode[user_id] = in_memory_episode

        logger.debug(
            "Seeded in-memory state for user {} from DB episode {} ({} messages)",
            user_id,
            episode.id,
            len(messages),
        )
        return True

    async def _evaluate_switch(
        self,
        user_id: int,
        content: str,
    ) -> SwitchDecision:
        """Evaluate whether to start a new episode."""

        # Check rate limiting first
        if self._get_recent_switch_count(user_id) >= self.config.max_episodes_per_hour:
            return SwitchDecision(
                should_switch=False,
                reason="Rate limit exceeded",
                confidence=1.0,
                trigger_type="rate_limit",
            )

        # Check if we need to seed in-memory state from DB
        current_in_memory = self._in_memory_manager._current_episode.get(user_id)
        if current_in_memory is None and self.db:
            # Try to seed state from DB to prevent unnecessary splits
            await self._seed_in_memory_state_from_db(user_id)

        # Get last message time from database
        if self.db:
            episode = await self.db.get_active_episode_for_user(user_id)
            if episode:
                # Check minimum episode duration
                if hasattr(episode, "started_at") and episode.started_at:
                    elapsed = (datetime.now() - episode.started_at).total_seconds()
                    if elapsed < self.config.min_episode_duration:
                        return SwitchDecision(
                            should_switch=False,
                            reason=f"Episode too young ({elapsed:.0f}s)",
                            confidence=1.0,
                            trigger_type="anti_flap_min_duration",
                        )

                # Check time gap
                if hasattr(episode, "last_user_message_at") and episode.last_user_message_at:
                    gap = (datetime.now() - episode.last_user_message_at).total_seconds()
                    if gap >= self.config.time_gap_threshold:
                        return SwitchDecision(
                            should_switch=True,
                            reason=f"Time gap: {gap / 3600:.1f} hours",
                            confidence=min(1.0, gap / self.config.time_gap_threshold),
                            trigger_type="time_gap",
                        )

                # We have an active DB episode and no time-gap trigger.
                # Don't start a new episode just because in-memory state is empty.
                # The in-memory state has been seeded above, so continue in same episode.
                if current_in_memory is None or not current_in_memory.messages:
                    return SwitchDecision(
                        should_switch=False,
                        reason="Continuing active DB episode",
                        confidence=1.0,
                        trigger_type=None,
                    )

        # Fall back to in-memory evaluation
        return await self._in_memory_manager.evaluate_switch(user_id, content)

    async def process_user_message(
        self,
        user_id: int,
        content: str,
    ) -> MessageResult:
        """Process a user message with episode management.

        Args:
            user_id: Telegram user ID
            content: Message content

        Returns:
            MessageResult with message, episode, and switch info
        """
        if self.db is None:
            raise RuntimeError("Database client not configured")

        # Evaluate episode switch
        switch_decision = await self._evaluate_switch(user_id, content)

        # Get or create thread
        thread = await self.db.get_or_create_thread(user_id)

        # Handle episode switch
        if switch_decision.should_switch:
            episode = await self.db.start_new_episode(thread.id)
            self._record_switch(user_id)
            is_new_episode = True
            logger.info(
                "Started new episode {} for user {} (reason: {})",
                episode.id,
                user_id,
                switch_decision.reason,
            )
        else:
            episode = await self.db.get_active_episode_for_user(user_id)
            if episode is None:
                episode = await self.db.start_new_episode(thread.id)
                is_new_episode = True
            else:
                is_new_episode = False

        # Persist message
        message = await self.db.add_message(
            telegram_user_id=user_id,
            role="user",
            content_text=content,
        )

        # Track in in-memory manager for topic detection
        await self._in_memory_manager.add_message(
            user_id=user_id,
            content=content,
            role="user",
        )

        return MessageResult(
            message=message,
            episode=episode,
            is_new_episode=is_new_episode,
            switch_decision=switch_decision,
        )

    async def process_assistant_message(
        self,
        user_id: int,
        content: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        model: str | None = None,
    ) -> MessageResult:
        """Process an assistant message.

        Args:
            user_id: Telegram user ID
            content: Message content
            tokens_in: Input token count
            tokens_out: Output token count
            model: Model name

        Returns:
            MessageResult with message and episode info
        """
        if self.db is None:
            raise RuntimeError("Database client not configured")

        # Get or create thread and episode
        thread = await self.db.get_or_create_thread(user_id)
        episode = await self.db.get_active_episode_for_user(user_id)
        if episode is None:
            episode = await self.db.start_new_episode(thread.id)
            is_new_episode = True
        else:
            is_new_episode = False

        # Persist message
        message = await self.db.add_message(
            telegram_user_id=user_id,
            role="assistant",
            content_text=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
        )

        return MessageResult(
            message=message,
            episode=episode,
            is_new_episode=is_new_episode,
            switch_decision=SwitchDecision(
                should_switch=False,
                reason="Assistant message",
                confidence=1.0,
            ),
        )

    async def close_current_episode(
        self,
        user_id: int,
        final_summary: str | None = None,
    ) -> "Episode | None":
        """Close the current episode for a user.

        Args:
            user_id: Telegram user ID
            final_summary: Optional summary to store

        Returns:
            Closed Episode or None if no active episode
        """
        if self.db is None:
            return None

        episode = await self.db.get_active_episode_for_user(user_id)
        if episode is None:
            return None

        closed = await self.db.close_episode(episode.id)

        if final_summary:
            # Store summary would go here
            pass

        logger.info("Closed episode {} for user {}", episode.id, user_id)
        return closed

    async def get_current_episode(self, user_id: int) -> "Episode | None":
        """Get the current episode for a user.

        Args:
            user_id: Telegram user ID

        Returns:
            Current Episode or None
        """
        if self.db is None:
            return None
        return await self.db.get_active_episode_for_user(user_id)

    async def get_episode_history(self, user_id: int) -> "list[Episode]":
        """Get episode history for a user.

        Args:
            user_id: Telegram user ID

        Returns:
            List of Episodes
        """
        if self.db is None:
            return []

        thread = await self.db.get_thread_for_user(user_id)
        if thread is None:
            return []

        return await self.db.get_episodes_for_thread(thread.id)

    async def get_messages_for_current_episode(
        self, user_id: int, limit: int = 100
    ) -> "list[EpisodeMessage]":
        """Get messages for the current episode.

        Args:
            user_id: Telegram user ID
            limit: Maximum number of messages

        Returns:
            List of EpisodeMessages
        """
        if self.db is None:
            return []

        episode = await self.db.get_active_episode_for_user(user_id)
        if episode is None:
            return []

        return await self.db.get_messages_for_episode(episode.id, limit)

    async def get_recent_messages(self, user_id: int, limit: int = 50) -> "list[EpisodeMessage]":
        """Get recent messages for a user.

        Args:
            user_id: Telegram user ID
            limit: Maximum number of messages

        Returns:
            List of EpisodeMessages
        """
        if self.db is None:
            return []
        return await self.db.get_recent_messages(user_id, limit)


# Global instance for dependency injection
_episode_manager: EpisodeManager | None = None


def get_episode_manager() -> EpisodeManager:
    """Get or create global episode manager instance."""
    global _episode_manager
    if _episode_manager is None:
        _episode_manager = EpisodeManager()
    return _episode_manager


def set_episode_manager(manager: EpisodeManager | None) -> None:
    """Set global episode manager instance (useful for testing)."""
    global _episode_manager
    _episode_manager = manager
