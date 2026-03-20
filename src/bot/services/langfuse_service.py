"""Langfuse observability service for LLM tracing and cost tracking."""

from __future__ import annotations

from typing import Any

from loguru import logger

from bot.config import settings

try:
    from langfuse.callback import CallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = None  # type: ignore[assignment, misc]


class LangfuseService:
    """Manages Langfuse tracing via per-request LangChain callback handlers.

    Gracefully unavailable if langfuse package not installed or keys not configured.
    """

    def __init__(self) -> None:
        self._available = False

        if not LANGFUSE_AVAILABLE:
            logger.info("LangfuseService: langfuse package not installed")
            return

        if not settings.langfuse_enabled:
            logger.info("LangfuseService: disabled via LANGFUSE_ENABLED=false")
            return

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.info("LangfuseService: keys not configured, tracing disabled")
            return

        self._available = True
        logger.info("LangfuseService initialized (host={})", settings.langfuse_base_url)

    @property
    def available(self) -> bool:
        return self._available

    def create_handler(
        self,
        user_id: int,
        session_id: str | None = None,
        trace_name: str = "chat",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any | None:
        """Create a per-request CallbackHandler for LangChain.

        Returns:
            CallbackHandler instance or None if unavailable.
        """
        if not self._available or CallbackHandler is None:
            return None

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
            user_id=str(user_id),
            session_id=session_id,
            trace_name=trace_name,
            tags=tags or [],
            metadata=metadata or {},
        )

    def create_config(
        self,
        user_id: int,
        session_id: str | None = None,
        trace_name: str = "chat",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a config dict for LangChain .ainvoke(config=...).

        Returns empty dict if unavailable (safe to pass to ainvoke).
        """
        handler = self.create_handler(user_id, session_id, trace_name, tags, metadata)
        if handler is None:
            return {}
        return {"callbacks": [handler]}

    def flush(self) -> None:
        """Flush pending events. Call on shutdown."""
        if not self._available:
            return
        # Each CallbackHandler flushes on __del__, but we can force it
        # by importing and flushing the global client if available
        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_base_url,
            )
            client.flush()
            client.shutdown()
        except Exception as e:
            logger.warning("Langfuse flush failed: {}", e)


# DI helpers
_langfuse_service: LangfuseService | None = None


def get_langfuse_service() -> LangfuseService:
    """Get or create global LangfuseService instance."""
    global _langfuse_service
    if _langfuse_service is None:
        _langfuse_service = LangfuseService()
    return _langfuse_service


def set_langfuse_service(service: LangfuseService | None) -> None:
    """Set global LangfuseService instance (useful for testing)."""
    global _langfuse_service
    _langfuse_service = service
