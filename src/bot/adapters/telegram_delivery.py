"""Delivery adapters implementing MessageDeliveryPort for different channels."""

from aiogram import Bot
from aiogram.types import BufferedInputFile

from bot.ports import MessageDeliveryPort


class TelegramDelivery:
    """Implements MessageDeliveryPort using aiogram Bot.

    This is the only place that imports aiogram for message delivery,
    keeping all other services framework-agnostic.
    """

    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_text(self, chat_id: int, text: str) -> None:
        """Send a text message via Telegram."""
        await self._bot.send_message(chat_id=chat_id, text=text)

    async def send_photo(
        self, chat_id: int, photo_bytes: bytes, caption: str | None = None
    ) -> None:
        """Send a photo via Telegram."""
        await self._bot.send_photo(
            chat_id=chat_id,
            photo=BufferedInputFile(photo_bytes, filename="photo.png"),
            caption=caption,
        )


# Runtime check: TelegramDelivery satisfies the protocol
assert isinstance(TelegramDelivery.__new__(TelegramDelivery), MessageDeliveryPort)
