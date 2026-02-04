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
    await dp.start_polling(bot)
