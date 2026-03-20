import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import settings
from bot.handlers import router


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

    # Start proactive scheduler (best-effort)
    scheduler = None
    try:
        from bot.services.proactive_scheduler import ProactiveScheduler, set_proactive_scheduler

        from bot.adapters import TelegramDelivery

        delivery = TelegramDelivery(bot)
        scheduler = ProactiveScheduler(delivery)
        scheduler.start()
        set_proactive_scheduler(scheduler)
    except Exception as e:
        from loguru import logger

        logger.warning("Proactive scheduler failed to start: {}", e)

    try:
        await dp.start_polling(bot)
    finally:
        if scheduler is not None:
            scheduler.stop()
        await bot.session.close()
        try:
            from bot.infra.langfuse_service import get_langfuse_service

            get_langfuse_service().flush()
        except Exception:
            pass
