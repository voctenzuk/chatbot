"""Tests for bot.wiring — composition root."""

import asyncio
from unittest.mock import MagicMock, patch

from bot.wiring import AppContext, build_app_context

# Patch paths — imports happen inside build_app_context, so we patch at source
_P_LLM = "bot.llm.service.LLMService"
_P_CTX = "bot.conversation.context_builder.ContextBuilder"
_P_LF = "bot.infra.langfuse_service.LangfuseService"
_P_EM = "bot.conversation.episode_manager.EpisodeManager"
_P_DB = "bot.infra.db_client.DatabaseClient"
_P_MEM = "bot.memory.cognee_service.CogneeMemoryService"
_P_IMG = "bot.media.image_service.ImageService"


def _patch_required(**overrides: MagicMock):
    """Context manager that patches all required + optional services.

    By default optional services raise (unavailable). Pass overrides
    to make specific optional services available.
    """
    from contextlib import ExitStack

    defaults = {
        _P_LLM: MagicMock(),
        _P_CTX: MagicMock(),
        _P_LF: MagicMock(),
        _P_EM: MagicMock(),
        _P_DB: RuntimeError("no db"),
        _P_MEM: RuntimeError("no cognee"),
        _P_IMG: RuntimeError("no openai"),
    }
    defaults.update(overrides)

    stack = ExitStack()
    mocks = {}
    for path, value in defaults.items():
        if isinstance(value, Exception):
            mocks[path] = stack.enter_context(patch(path, side_effect=value))
        elif isinstance(value, MagicMock):
            mocks[path] = stack.enter_context(patch(path, return_value=value))
        else:
            mocks[path] = stack.enter_context(patch(path, value))
    return stack, mocks


class TestAppContext:
    """Tests for AppContext dataclass and shutdown."""

    async def test_shutdown_awaits_background_tasks(self) -> None:
        completed = False

        async def bg_task() -> None:
            nonlocal completed
            await asyncio.sleep(0.01)
            completed = True

        pipeline = MagicMock()
        task = asyncio.create_task(bg_task())
        pipeline._background_tasks = {task}

        ctx = AppContext(
            llm=MagicMock(),
            episode_manager=MagicMock(),
            context_builder=MagicMock(),
            langfuse=MagicMock(),
            pipeline=pipeline,
        )
        await ctx.shutdown(timeout=5.0)
        assert completed is True

    async def test_shutdown_cancels_on_timeout(self) -> None:
        async def forever() -> None:
            await asyncio.sleep(999)

        pipeline = MagicMock()
        task = asyncio.create_task(forever())
        pipeline._background_tasks = {task}

        ctx = AppContext(
            llm=MagicMock(),
            episode_manager=MagicMock(),
            context_builder=MagicMock(),
            langfuse=MagicMock(),
            pipeline=pipeline,
        )
        await ctx.shutdown(timeout=0.05)
        # Give event loop a tick to process cancellation
        await asyncio.sleep(0.01)
        assert task.cancelled()

    async def test_shutdown_stops_scheduler(self) -> None:
        scheduler = MagicMock()
        ctx = AppContext(
            llm=MagicMock(),
            episode_manager=MagicMock(),
            context_builder=MagicMock(),
            langfuse=MagicMock(),
            scheduler=scheduler,
        )
        await ctx.shutdown()
        scheduler.stop.assert_called_once()

    async def test_shutdown_flushes_langfuse_sync(self) -> None:
        langfuse = MagicMock()
        ctx = AppContext(
            llm=MagicMock(),
            episode_manager=MagicMock(),
            context_builder=MagicMock(),
            langfuse=langfuse,
        )
        await ctx.shutdown()
        langfuse.flush.assert_called_once()

    async def test_shutdown_no_pipeline(self) -> None:
        ctx = AppContext(
            llm=MagicMock(),
            episode_manager=MagicMock(),
            context_builder=MagicMock(),
            langfuse=MagicMock(),
            pipeline=None,
        )
        await ctx.shutdown()  # should not raise

    async def test_shutdown_scheduler_error_swallowed(self) -> None:
        scheduler = MagicMock()
        scheduler.stop.side_effect = RuntimeError("stop failed")
        ctx = AppContext(
            llm=MagicMock(),
            episode_manager=MagicMock(),
            context_builder=MagicMock(),
            langfuse=MagicMock(),
            scheduler=scheduler,
        )
        await ctx.shutdown()  # should not raise


class TestBuildAppContext:
    """Tests for build_app_context factory."""

    async def test_all_required_services_created(self) -> None:
        mock_llm = MagicMock()
        mock_ctx = MagicMock()
        mock_lf = MagicMock()
        mock_em = MagicMock()

        with (
            patch(_P_LLM, return_value=mock_llm),
            patch(_P_CTX, return_value=mock_ctx),
            patch(_P_LF, return_value=mock_lf),
            patch(_P_EM, return_value=mock_em),
            patch(_P_DB, side_effect=RuntimeError("no db")),
            patch(_P_MEM, side_effect=RuntimeError("no cognee")),
            patch(_P_IMG, side_effect=RuntimeError("no openai")),
        ):
            ctx = await build_app_context()

        assert ctx.llm is mock_llm
        assert ctx.episode_manager is mock_em
        assert ctx.context_builder is mock_ctx
        assert ctx.langfuse is mock_lf
        assert ctx.memory is None
        assert ctx.db_client is None
        assert ctx.image_service is None
        assert ctx.pipeline is not None

    async def test_all_optional_services_available(self) -> None:
        mock_db = MagicMock()
        mock_mem = MagicMock()
        mock_img = MagicMock()

        with (
            patch(_P_LLM, return_value=MagicMock()),
            patch(_P_CTX, return_value=MagicMock()),
            patch(_P_LF, return_value=MagicMock()),
            patch(_P_EM, return_value=MagicMock()),
            patch(_P_DB, return_value=mock_db),
            patch(_P_MEM, return_value=mock_mem),
            patch(_P_IMG, return_value=mock_img),
        ):
            ctx = await build_app_context()

        assert ctx.memory is mock_mem
        assert ctx.db_client is mock_db
        assert ctx.image_service is mock_img
        assert ctx.pipeline is not None

    async def test_memory_unavailable(self) -> None:
        with (
            patch(_P_LLM, return_value=MagicMock()),
            patch(_P_CTX, return_value=MagicMock()),
            patch(_P_LF, return_value=MagicMock()),
            patch(_P_EM, return_value=MagicMock()),
            patch(_P_DB, side_effect=RuntimeError("no db")),
            patch(_P_MEM, side_effect=ImportError("no cognee")),
            patch(_P_IMG, side_effect=RuntimeError("no openai")),
        ):
            ctx = await build_app_context()
        assert ctx.memory is None

    async def test_db_unavailable_episode_manager_gets_none(self) -> None:
        mock_em_cls = MagicMock()
        with (
            patch(_P_LLM, return_value=MagicMock()),
            patch(_P_CTX, return_value=MagicMock()),
            patch(_P_LF, return_value=MagicMock()),
            patch(_P_EM, mock_em_cls),
            patch(_P_DB, side_effect=RuntimeError("no supabase")),
            patch(_P_MEM, side_effect=RuntimeError("no cognee")),
            patch(_P_IMG, side_effect=RuntimeError("no openai")),
        ):
            ctx = await build_app_context()
        assert ctx.db_client is None
        mock_em_cls.assert_called_once_with(db_client=None)

    async def test_image_unavailable(self) -> None:
        with (
            patch(_P_LLM, return_value=MagicMock()),
            patch(_P_CTX, return_value=MagicMock()),
            patch(_P_LF, return_value=MagicMock()),
            patch(_P_EM, return_value=MagicMock()),
            patch(_P_DB, side_effect=RuntimeError("no db")),
            patch(_P_MEM, side_effect=RuntimeError("no cognee")),
            patch(_P_IMG, side_effect=ImportError("no openai")),
        ):
            ctx = await build_app_context()
        assert ctx.image_service is None

    async def test_langfuse_failure_creates_stub(self) -> None:
        with (
            patch(_P_LLM, return_value=MagicMock()),
            patch(_P_CTX, return_value=MagicMock()),
            patch(_P_LF, side_effect=RuntimeError("no langfuse")),
            patch(_P_EM, return_value=MagicMock()),
            patch(_P_DB, side_effect=RuntimeError("no db")),
            patch(_P_MEM, side_effect=RuntimeError("no cognee")),
            patch(_P_IMG, side_effect=RuntimeError("no openai")),
        ):
            ctx = await build_app_context()
        assert ctx.langfuse is not None
        assert ctx.langfuse.create_config() == {}
        ctx.langfuse.flush()  # should not raise

    async def test_episode_manager_gets_db_client(self) -> None:
        mock_db = MagicMock()
        mock_em_cls = MagicMock()
        with (
            patch(_P_LLM, return_value=MagicMock()),
            patch(_P_CTX, return_value=MagicMock()),
            patch(_P_LF, return_value=MagicMock()),
            patch(_P_EM, mock_em_cls),
            patch(_P_DB, return_value=mock_db),
            patch(_P_MEM, side_effect=RuntimeError("no cognee")),
            patch(_P_IMG, side_effect=RuntimeError("no openai")),
        ):
            await build_app_context()
        mock_em_cls.assert_called_once_with(db_client=mock_db)
