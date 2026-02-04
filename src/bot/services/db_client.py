"""Database client for Supabase integration.

This module provides a Supabase client for interacting with the database,
including the memory system tables (threads, episodes, messages, summaries).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from unittest.mock import MagicMock

from loguru import logger


class DatabaseInitializationError(RuntimeError):
    """Raised when database client initialization fails.

    This exception indicates that the database is unavailable and
    message persistence cannot be guaranteed. Handlers should catch
    this and inform users appropriately.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


# Import supabase client - may not be available in all environments
try:
    from supabase import Client, create_client

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None  # type: ignore[misc, assignment]
    create_client = None  # type: ignore[misc, assignment]


class DatabaseClientProtocol(Protocol):
    """Protocol for database client to enable mocking/testing."""

    def rpc(self, *args: Any, **kwargs: Any) -> Any:
        """Call a stored procedure."""
        ...

    def table(self, table_name: str) -> Any:
        """Get a table reference."""
        ...


@dataclass
class Thread:
    """A conversation thread for a user."""

    id: str
    telegram_user_id: int
    active_episode_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Thread:
        """Create Thread from database row."""
        return cls(
            id=str(row["id"]),
            telegram_user_id=row["telegram_user_id"],
            active_episode_id=str(row["active_episode_id"])
            if row.get("active_episode_id")
            else None,
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            if row.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
            if row.get("updated_at")
            else None,
        )


@dataclass
class Episode:
    """A conversation episode (session/chapter) within a thread."""

    id: str
    thread_id: str
    status: str  # 'active' or 'closed'
    started_at: datetime
    ended_at: datetime | None = None
    topic_label: str | None = None
    last_user_message_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Episode:
        """Create Episode from database row."""
        return cls(
            id=str(row["id"]),
            thread_id=str(row["thread_id"]),
            status=row["status"],
            started_at=datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
            if row.get("started_at")
            else datetime.now(),
            ended_at=datetime.fromisoformat(row["ended_at"].replace("Z", "+00:00"))
            if row.get("ended_at")
            else None,
            topic_label=row.get("topic_label"),
            last_user_message_at=datetime.fromisoformat(
                row["last_user_message_at"].replace("Z", "+00:00")
            )
            if row.get("last_user_message_at")
            else None,
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            if row.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
            if row.get("updated_at")
            else None,
        )

    @property
    def is_active(self) -> bool:
        """Check if episode is active."""
        return self.status == "active"


@dataclass
class EpisodeMessage:
    """A message within an episode."""

    id: str
    episode_id: str
    role: str  # 'user', 'assistant', 'system'
    content_text: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str | None = None
    created_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> EpisodeMessage:
        """Create EpisodeMessage from database row."""
        return cls(
            id=str(row["id"]),
            episode_id=str(row["episode_id"]),
            role=row["role"],
            content_text=row["content_text"],
            tokens_in=row.get("tokens_in"),
            tokens_out=row.get("tokens_out"),
            model=row.get("model"),
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            if row.get("created_at")
            else None,
        )


@dataclass
class EpisodeSummary:
    """A summary of an episode."""

    id: str
    episode_id: str
    kind: str  # 'running', 'chunk', 'final'
    summary_text: str
    summary_json: dict[str, Any] | None = None
    created_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> EpisodeSummary:
        """Create EpisodeSummary from database row."""
        return cls(
            id=str(row["id"]),
            episode_id=str(row["episode_id"]),
            kind=row["kind"],
            summary_text=row["summary_text"],
            summary_json=row.get("summary_json"),
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            if row.get("created_at")
            else None,
        )


class DatabaseClient:
    """Database client for Supabase integration.

    Provides methods for interacting with the memory system tables:
    - threads: User conversation threads
    - episodes: Session/chapters within threads
    - messages: Individual messages within episodes
    - episode_summaries: Summaries of episodes
    """

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        client: DatabaseClientProtocol | None = None,
    ) -> None:
        """Initialize database client.

        Args:
            url: Supabase URL. If not provided, reads from SUPABASE_URL env var.
            key: Supabase service role key. If not provided, reads from SUPABASE_SERVICE_KEY env var.
            client: Optional pre-configured client (for testing/mocking).

        Raises:
            RuntimeError: If supabase package is not installed or credentials are missing.
        """
        self._client: DatabaseClientProtocol
        self._is_mock = False

        if client is not None:
            # Use provided client (for testing)
            self._client = client
            self._is_mock = isinstance(client, MagicMock)
            logger.info("DatabaseClient initialized with custom client")
            return

        if not SUPABASE_AVAILABLE:
            raise RuntimeError(
                "supabase package is not installed. Install it with: pip install supabase"
            )

        # Get configuration from parameters or env vars
        self._url = url or os.getenv("SUPABASE_URL")
        self._key = key or os.getenv("SUPABASE_SERVICE_KEY")

        if not self._url or not self._key:
            raise RuntimeError(
                "Supabase URL and service key are required. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables."
            )

        # Initialize the Supabase client
        if create_client is None:
            raise RuntimeError("supabase package is not installed")
        self._client: DatabaseClientProtocol = create_client(self._url, self._key)
        logger.info("DatabaseClient initialized with Supabase")

    async def get_or_create_thread(self, telegram_user_id: int) -> Thread:
        """Get or create a thread for a user.

        Args:
            telegram_user_id: Telegram user ID.

        Returns:
            Thread object.
        """
        try:
            response = self._client.rpc(
                "get_or_create_thread",
                {"p_telegram_user_id": telegram_user_id},
            ).execute()

            # The function returns a UUID, we need to fetch the thread
            thread_id = response.data
            return await self.get_thread_by_id(thread_id)
        except Exception as e:
            logger.error("Failed to get or create thread for user {}: {}", telegram_user_id, e)
            raise

    async def get_thread_by_id(self, thread_id: str) -> Thread:
        """Get a thread by ID.

        Args:
            thread_id: Thread UUID.

        Returns:
            Thread object.
        """
        try:
            response = (
                self._client.table("threads").select("*").eq("id", thread_id).single().execute()
            )
            return Thread.from_row(response.data)
        except Exception as e:
            logger.error("Failed to get thread {}: {}", thread_id, e)
            raise

    async def get_thread_for_user(self, telegram_user_id: int) -> Thread | None:
        """Get the thread for a user if it exists.

        Args:
            telegram_user_id: Telegram user ID.

        Returns:
            Thread object or None if not found.
        """
        try:
            response = (
                self._client.table("threads")
                .select("*")
                .eq("telegram_user_id", telegram_user_id)
                .maybe_single()
                .execute()
            )
            if response.data:
                return Thread.from_row(response.data)
            return None
        except Exception as e:
            logger.error("Failed to get thread for user {}: {}", telegram_user_id, e)
            return None

    async def start_new_episode(
        self,
        thread_id: str,
        topic_label: str | None = None,
    ) -> Episode:
        """Start a new episode for a thread.

        Args:
            thread_id: Thread UUID.
            topic_label: Optional topic label for the episode.

        Returns:
            New Episode object.
        """
        try:
            response = self._client.rpc(
                "start_new_episode",
                {
                    "p_thread_id": thread_id,
                    "p_topic_label": topic_label,
                },
            ).execute()

            episode_id = response.data
            return await self.get_episode_by_id(episode_id)
        except Exception as e:
            logger.error("Failed to start new episode for thread {}: {}", thread_id, e)
            raise

    async def get_episode_by_id(self, episode_id: str) -> Episode:
        """Get an episode by ID.

        Args:
            episode_id: Episode UUID.

        Returns:
            Episode object.
        """
        try:
            response = (
                self._client.table("episodes").select("*").eq("id", episode_id).single().execute()
            )
            return Episode.from_row(response.data)
        except Exception as e:
            logger.error("Failed to get episode {}: {}", episode_id, e)
            raise

    async def get_active_episode_for_user(self, telegram_user_id: int) -> Episode | None:
        """Get the active episode for a user.

        Args:
            telegram_user_id: Telegram user ID.

        Returns:
            Active Episode or None if not found.
        """
        try:
            thread = await self.get_thread_for_user(telegram_user_id)
            if not thread or not thread.active_episode_id:
                return None
            return await self.get_episode_by_id(thread.active_episode_id)
        except Exception as e:
            logger.error("Failed to get active episode for user {}: {}", telegram_user_id, e)
            return None

    async def close_episode(self, episode_id: str) -> Episode:
        """Close an episode.

        Args:
            episode_id: Episode UUID to close.

        Returns:
            Updated Episode object.
        """
        try:
            response = (
                self._client.table("episodes")
                .update(
                    {
                        "status": "closed",
                        "ended_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    }
                )
                .eq("id", episode_id)
                .execute()
            )
            return Episode.from_row(response.data[0])
        except Exception as e:
            logger.error("Failed to close episode {}: {}", episode_id, e)
            raise

    async def add_message(
        self,
        telegram_user_id: int,
        role: str,
        content_text: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        model: str | None = None,
    ) -> EpisodeMessage:
        """Add a message to the current episode.

        Args:
            telegram_user_id: Telegram user ID.
            role: Message role ('user', 'assistant', 'system').
            content_text: Message content.
            tokens_in: Optional token count for input.
            tokens_out: Optional token count for output.
            model: Optional model name.

        Returns:
            Created EpisodeMessage.
        """
        try:
            response = self._client.rpc(
                "add_message_to_current_episode",
                {
                    "p_telegram_user_id": telegram_user_id,
                    "p_role": role,
                    "p_content_text": content_text,
                    "p_tokens_in": tokens_in,
                    "p_tokens_out": tokens_out,
                    "p_model": model,
                },
            ).execute()

            message_id = response.data
            return await self.get_message_by_id(message_id)
        except Exception as e:
            logger.error("Failed to add message for user {}: {}", telegram_user_id, e)
            raise

    async def get_message_by_id(self, message_id: str) -> EpisodeMessage:
        """Get a message by ID.

        Args:
            message_id: Message UUID.

        Returns:
            EpisodeMessage object.
        """
        try:
            response = (
                self._client.table("messages").select("*").eq("id", message_id).single().execute()
            )
            return EpisodeMessage.from_row(response.data)
        except Exception as e:
            logger.error("Failed to get message {}: {}", message_id, e)
            raise

    @staticmethod
    def _normalize_message_row(row: dict[str, Any]) -> dict[str, Any]:
        """Normalize RPC message row to match EpisodeMessage.from_row expectations.

        The get_recent_messages RPC returns 'message_id' but EpisodeMessage.from_row
        expects 'id'. This method normalizes the row by mapping message_id to id.

        Args:
            row: Raw row from RPC response.

        Returns:
            Normalized row with 'id' key.
        """
        normalized = dict(row)
        # RPC returns 'message_id', but from_row expects 'id'
        if "message_id" in normalized and "id" not in normalized:
            normalized["id"] = normalized.pop("message_id")
        return normalized

    async def get_recent_messages(
        self,
        telegram_user_id: int,
        limit: int = 50,
    ) -> list[EpisodeMessage]:
        """Get recent messages for a user.

        Args:
            telegram_user_id: Telegram user ID.
            limit: Maximum number of messages to return.

        Returns:
            List of EpisodeMessage objects, most recent first.
        """
        try:
            response = self._client.rpc(
                "get_recent_messages",
                {
                    "p_telegram_user_id": telegram_user_id,
                    "p_limit": limit,
                },
            ).execute()

            # Normalize rows: RPC returns 'message_id' but from_row expects 'id'
            normalized_rows = [self._normalize_message_row(row) for row in response.data]
            return [EpisodeMessage.from_row(row) for row in normalized_rows]
        except Exception as e:
            logger.error("Failed to get recent messages for user {}: {}", telegram_user_id, e)
            return []

    async def get_messages_for_episode(
        self,
        episode_id: str,
        limit: int = 100,
    ) -> list[EpisodeMessage]:
        """Get messages for a specific episode.

        Args:
            episode_id: Episode UUID.
            limit: Maximum number of messages to return.

        Returns:
            List of EpisodeMessage objects, oldest first.
        """
        try:
            response = (
                self._client.table("messages")
                .select("*")
                .eq("episode_id", episode_id)
                .order("created_at", desc=False)
                .limit(limit)
                .execute()
            )
            return [EpisodeMessage.from_row(row) for row in response.data]
        except Exception as e:
            logger.error("Failed to get messages for episode {}: {}", episode_id, e)
            return []

    async def get_episodes_for_thread(
        self,
        thread_id: str,
        status: str | None = None,
    ) -> list[Episode]:
        """Get episodes for a thread.

        Args:
            thread_id: Thread UUID.
            status: Optional status filter ('active' or 'closed').

        Returns:
            List of Episode objects.
        """
        try:
            query = self._client.table("episodes").select("*").eq("thread_id", thread_id)
            if status:
                query = query.eq("status", status)
            response = query.order("started_at", desc=False).execute()
            return [Episode.from_row(row) for row in response.data]
        except Exception as e:
            logger.error("Failed to get episodes for thread {}: {}", thread_id, e)
            return []

    async def upsert_episode_summary(
        self,
        episode_id: str,
        kind: str,
        summary_text: str,
        summary_json: dict[str, Any] | None = None,
    ) -> EpisodeSummary:
        """Upsert an episode summary.

        Args:
            episode_id: Episode UUID.
            kind: Summary kind ('running', 'chunk', 'final').
            summary_text: Summary text.
            summary_json: Optional structured summary data.

        Returns:
            Created/updated EpisodeSummary.
        """
        try:
            response = self._client.rpc(
                "upsert_episode_summary",
                {
                    "p_episode_id": episode_id,
                    "p_kind": kind,
                    "p_summary_text": summary_text,
                    "p_summary_json": summary_json,
                },
            ).execute()

            summary_id = response.data
            return await self.get_summary_by_id(summary_id)
        except Exception as e:
            logger.error("Failed to upsert summary for episode {}: {}", episode_id, e)
            raise

    async def get_summary_by_id(self, summary_id: str) -> EpisodeSummary:
        """Get a summary by ID.

        Args:
            summary_id: Summary UUID.

        Returns:
            EpisodeSummary object.
        """
        try:
            response = (
                self._client.table("episode_summaries")
                .select("*")
                .eq("id", summary_id)
                .single()
                .execute()
            )
            return EpisodeSummary.from_row(response.data)
        except Exception as e:
            logger.error("Failed to get summary {}: {}", summary_id, e)
            raise

    async def get_summaries_for_episode(
        self,
        episode_id: str,
        kind: str | None = None,
    ) -> list[EpisodeSummary]:
        """Get summaries for an episode.

        Args:
            episode_id: Episode UUID.
            kind: Optional kind filter.

        Returns:
            List of EpisodeSummary objects.
        """
        try:
            query = self._client.table("episode_summaries").select("*").eq("episode_id", episode_id)
            if kind:
                query = query.eq("kind", kind)
            response = query.order("created_at", desc=True).execute()
            return [EpisodeSummary.from_row(row) for row in response.data]
        except Exception as e:
            logger.error("Failed to get summaries for episode {}: {}", episode_id, e)
            return []


# Global instance for dependency injection
_db_client: DatabaseClient | None = None


def get_db_client() -> DatabaseClient:
    """Get or create global database client instance.

    Returns:
        DatabaseClient instance.

    Raises:
        RuntimeError: If client cannot be initialized.
    """
    global _db_client
    if _db_client is None:
        _db_client = DatabaseClient()
    return _db_client


def set_db_client(client: DatabaseClient | None) -> None:
    """Set global database client instance (useful for testing).

    Args:
        client: DatabaseClient instance or None to reset.
    """
    global _db_client
    _db_client = client
