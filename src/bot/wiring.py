"""Composition root — single place to create and wire all services.

build_app_context() creates every service once at startup, handling
graceful degradation for optional dependencies (mem0, Supabase, OpenAI).
AppContext groups all services so they can be passed around as a unit.
"""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class AppContext:
    """Holds all service instances created at startup.

    Optional services are None when their dependencies are unavailable.
    """

    llm: Any  # LLMService
    episode_manager: Any  # EpisodeManager
    context_builder: Any  # ContextBuilder
    langfuse: Any  # LangfuseService
    memory: Any | None = None  # Mem0MemoryService
    image_service: Any | None = None  # ImageService
    db_client: Any | None = None  # DatabaseClient
    scheduler: Any | None = None  # ProactiveScheduler
    character: Any | None = None  # CharacterConfig
    fact_extractor: Any | None = None  # FactExtractorService
    profile_builder: Any | None = None  # UserProfileBuilder
    relationship_scorer: Any | None = None  # RelationshipScorer
    pipeline: Any | None = None  # ChatPipeline (set after construction)
    _background_tasks: set[asyncio.Task[Any]] = field(default_factory=set, repr=False)

    async def shutdown(self, shutdown_timeout: float = 5.0) -> None:
        """Graceful shutdown: await background tasks, stop scheduler, flush langfuse."""
        # 1. Await pending background tasks from pipeline
        if self.pipeline is not None:
            pending = list(getattr(self.pipeline, "_background_tasks", set()))
            if pending:
                logger.info("Waiting for {} background tasks to complete...", len(pending))
                done, not_done = await asyncio.wait(pending, timeout=shutdown_timeout)
                for task in not_done:
                    task.cancel()
                if not_done:
                    logger.warning(
                        "Cancelled {} background tasks after {}s timeout",
                        len(not_done),
                        shutdown_timeout,
                    )

        # 2. Stop proactive scheduler
        if self.scheduler is not None:
            try:
                self.scheduler.stop()
            except Exception as exc:
                logger.warning("Scheduler stop failed: {}", exc)

        # 3. Flush langfuse (sync call — no await)
        if self.langfuse is not None:
            try:
                self.langfuse.flush()
            except Exception as exc:
                logger.warning("Langfuse flush failed: {}", exc)

        logger.info("AppContext shutdown complete")


async def build_app_context() -> AppContext:
    """Create all services with graceful degradation for optional ones.

    Services that fail to initialize (mem0, Supabase, OpenAI Images)
    are set to None. Required services (LLM, ContextBuilder, Langfuse,
    EpisodeManager) are always created.

    Returns:
        Populated AppContext with all services.
    """
    from bot.config import settings

    if not settings.llm_api_key:
        logger.error("LLM_API_KEY is required — bot cannot respond without it")
        raise SystemExit(1)

    # --- Required services ---
    from bot.llm.service import LLMService

    llm = LLMService()
    logger.info("LLMService created")

    from bot.conversation.context_builder import ContextBuilder

    context_builder = ContextBuilder()
    logger.info("ContextBuilder created")

    from bot.infra.langfuse_service import LangfuseService

    if settings.langfuse_enabled and settings.langfuse_public_key:
        try:
            langfuse = LangfuseService()
        except Exception as exc:
            logger.warning("LangfuseService init failed, creating stub: {}", exc)
            langfuse = _StubLangfuse()
    else:
        langfuse = _StubLangfuse()

    # --- Optional services ---
    db_client = None
    try:
        from bot.infra.db_client import DatabaseClient

        db_client = DatabaseClient()
        logger.info("DatabaseClient created")
    except Exception as exc:
        logger.info("DatabaseClient unavailable ({}), running without DB", exc)

    memory = None
    try:
        from bot.memory.mem0_service import Mem0MemoryService

        memory = Mem0MemoryService()
        logger.info("Mem0MemoryService created")
    except Exception as exc:
        logger.info("Mem0MemoryService unavailable ({}), running without memory", exc)

    from bot.character import DEFAULT_CHARACTER

    image_service = None
    try:
        from bot.media.image_service import ImageService

        image_service = ImageService(character=DEFAULT_CHARACTER)
        logger.info(
            "ImageService created (reference_image={})",
            bool(DEFAULT_CHARACTER.reference_image_url),
        )
    except Exception as exc:
        logger.info("ImageService unavailable ({}), running without image generation", exc)

    # --- EpisodeManager (works with or without DB) ---
    from bot.conversation.episode_manager import EpisodeManager

    episode_manager = EpisodeManager(db_client=db_client)
    logger.info("EpisodeManager created (db={})", "yes" if db_client else "no")

    # --- Relationship scorer (optional — gracefully degrades) ---
    relationship_scorer = None
    try:
        from bot.memory.relationship_scorer import RelationshipScorer

        relationship_scorer = RelationshipScorer()
        logger.info("RelationshipScorer created")
    except Exception as exc:
        logger.info("RelationshipScorer unavailable ({}), skipping", exc)

    # --- Fact extraction pipeline (optional — requires LLM) ---
    fact_extractor = None
    profile_builder = None
    try:
        from bot.memory.fact_extractor import FactExtractorService
        from bot.memory.profile_builder import UserProfileBuilder

        fact_extractor = FactExtractorService(llm=llm, mem0_service=memory)
        profile_builder = UserProfileBuilder(
            db_client=db_client,
            relationship_scorer=relationship_scorer,
        )
        logger.info("FactExtractorService and UserProfileBuilder created")
    except Exception as exc:
        logger.info("Fact extraction pipeline unavailable ({}), skipping", exc)

    # --- ChatPipeline ---
    from bot.chat_pipeline import ChatPipeline

    pipeline = ChatPipeline(
        llm=llm,
        episode_manager=episode_manager,
        context_builder=context_builder,
        langfuse=langfuse,
        memory=memory,
        image_service=image_service,
        db_client=db_client,
        character=DEFAULT_CHARACTER,
        fact_extractor=fact_extractor,
        profile_builder=profile_builder,
        relationship_scorer=relationship_scorer,
    )
    logger.info("ChatPipeline created")

    ctx = AppContext(
        llm=llm,
        episode_manager=episode_manager,
        context_builder=context_builder,
        langfuse=langfuse,
        memory=memory,
        image_service=image_service,
        db_client=db_client,
        character=DEFAULT_CHARACTER,
        fact_extractor=fact_extractor,
        profile_builder=profile_builder,
        relationship_scorer=relationship_scorer,
        pipeline=pipeline,
    )

    logger.info(
        "AppContext built: memory={}, db={}, images={}, fact_extractor={}, relationship_scorer={}",
        memory is not None,
        db_client is not None,
        image_service is not None,
        fact_extractor is not None,
        relationship_scorer is not None,
    )
    return ctx


class _StubLangfuse:
    """No-op stub when Langfuse is unavailable."""

    @contextmanager
    def trace(self, **kwargs: Any) -> Iterator[None]:
        yield

    def flush(self) -> None:
        pass
