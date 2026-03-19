import asyncio

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode

from bot.config import settings
from bot.handlers import router


def run() -> None:
    asyncio.run(_amain())


async def _amain() -> None:
    assert settings.telegram_bot_token is not None, "TELEGRAM_BOT_TOKEN not set"
    bot = Bot(token=settings.telegram_bot_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()
    dp.include_router(router)

    # Start proactive scheduler (best-effort)
    scheduler = None
    try:
        from bot.services.proactive_scheduler import ProactiveScheduler, set_proactive_scheduler

        scheduler = ProactiveScheduler(bot)
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
