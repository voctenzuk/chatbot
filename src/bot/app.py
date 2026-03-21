import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from loguru import logger

from bot.config import settings
from bot.handlers import router
from bot.wiring import build_app_context


def run() -> None:
    asyncio.run(_amain())


async def _amain() -> None:
    if settings.telegram_bot_token is None:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    bot = Bot(
        token=settings.telegram_bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    dp.include_router(router)

    # Build composition root — creates all services once
    ctx = await build_app_context()

    # Inject dependencies into dispatcher workflow_data (aiogram kwargs injection)
    dp["pipeline"] = ctx.pipeline
    dp["db_client"] = ctx.db_client

    # Bridge composed services to legacy singletons for ProactiveScheduler
    # (it still calls get_*() internally — P3 TODO to refactor)
    from bot.conversation.episode_manager import set_episode_manager
    from bot.infra.langfuse_service import set_langfuse_service
    from bot.llm.service import set_llm_service

    set_llm_service(ctx.llm)
    set_episode_manager(ctx.episode_manager)
    set_langfuse_service(ctx.langfuse)
    if ctx.db_client is not None:
        from bot.infra.db_client import set_db_client

        set_db_client(ctx.db_client)

    # Start proactive scheduler (best-effort, needs bot for TelegramDelivery)
    try:
        from bot.adapters import TelegramDelivery
        from bot.adapters.proactive_scheduler import ProactiveScheduler, set_proactive_scheduler

        delivery = TelegramDelivery(bot)
        scheduler = ProactiveScheduler(delivery)
        scheduler.start()
        set_proactive_scheduler(scheduler)
        ctx.scheduler = scheduler
    except Exception as e:
        logger.warning("Proactive scheduler failed to start: {}", e)

    try:
        await dp.start_polling(bot)
    finally:
        await ctx.shutdown()
        await bot.session.close()
