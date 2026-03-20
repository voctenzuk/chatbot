"""Langfuse observability service for LLM tracing and cost tracking."""

from __future__ import annotations

from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from loguru import logger

from bot.config import settings


class LangfuseService:
    """Manages Langfuse tracing via per-request LangChain callback handlers."""

    def __init__(self) -> None:
        self._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
        logger.info("LangfuseService initialized (host={})", settings.langfuse_base_url)

    def create_handler(
        self,
        user_id: int,
        session_id: str | None = None,
        trace_name: str = "chat",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CallbackHandler:
        """Create a per-request CallbackHandler for LangChain."""
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
        """Create a config dict for LangChain .ainvoke(config=...)."""
        handler = self.create_handler(user_id, session_id, trace_name, tags, metadata)
        return {"callbacks": [handler]}

    def flush(self) -> None:
        """Flush pending events. Call on shutdown."""
        self._client.flush()
        self._client.shutdown()


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
