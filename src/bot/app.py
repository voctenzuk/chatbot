import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from loguru import logger

from bot.adapters import TelegramDelivery
from bot.adapters.proactive_scheduler import ProactiveScheduler, set_proactive_scheduler
from bot.config import settings
from bot.conversation.episode_manager import set_episode_manager
from bot.handlers import router
from bot.infra.db_client import set_db_client
from bot.infra.langfuse_service import set_langfuse_service
from bot.llm.service import set_llm_service
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

    # Bridge composed services to singletons used by infra modules
    set_llm_service(ctx.llm)
    set_episode_manager(ctx.episode_manager)
    set_langfuse_service(ctx.langfuse)
    if ctx.db_client is not None:
        set_db_client(ctx.db_client)

    # Start proactive scheduler with full dependency injection (needs bot for delivery)
    try:
        delivery = TelegramDelivery(bot)
        scheduler = ProactiveScheduler(
            delivery=delivery,
            llm=ctx.llm,
            db_client=ctx.db_client,
            episode_manager=ctx.episode_manager,
            relationship_scorer=ctx.relationship_scorer,
            profile_builder=ctx.profile_builder,
            character=ctx.character,
        )
        scheduler.start()
        set_proactive_scheduler(scheduler)
        ctx.scheduler = scheduler

        # Wire milestone callback: breaks the circular dep by setting after both exist.
        if ctx.relationship_scorer is not None:
            ctx.relationship_scorer.set_milestone_callback(scheduler.send_milestone_message)

        logger.info("ProactiveScheduler started with full dependency injection")
    except Exception as e:
        logger.warning("Proactive scheduler failed to start: {}", e)

    try:
        await dp.start_polling(bot)
    finally:
        await ctx.shutdown()
        await bot.session.close()
