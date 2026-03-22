"""Langfuse observability service for LLM tracing and cost tracking.

Langfuse v4 uses OpenTelemetry-based auto-instrumentation for LangChain.
Per-request context (user_id, session_id) is propagated via
``langfuse.propagate_attributes`` which sets OTel span attributes
on all spans created within the context manager scope.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from langfuse import Langfuse, propagate_attributes
from loguru import logger

from bot.config import settings


class LangfuseService:
    """Manages Langfuse tracing via OTel auto-instrumentation (v4+)."""

    def __init__(self) -> None:
        self._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
        logger.info("LangfuseService initialized (host={})", settings.langfuse_base_url)

    @contextmanager
    def trace(
        self,
        *,
        user_id: int,
        session_id: str | None = None,
        trace_name: str = "chat",
        tags: list[str] | None = None,
    ) -> Iterator[None]:
        """Set per-request OTel attributes for all spans in this scope."""
        with propagate_attributes(
            user_id=str(user_id),
            session_id=session_id,
            trace_name=trace_name,
            tags=tags or [],
        ):
            yield

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
