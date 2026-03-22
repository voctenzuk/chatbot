"""Tests for delivery adapters."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.adapters import TelegramDelivery
from bot.ports import MessageDeliveryPort


class TestTelegramDelivery:
    """Test TelegramDelivery implements MessageDeliveryPort correctly."""

    def test_satisfies_protocol(self) -> None:
        mock_bot = MagicMock()
        delivery = TelegramDelivery(bot=mock_bot)
        assert isinstance(delivery, MessageDeliveryPort)

    @pytest.mark.asyncio
    async def test_send_text_calls_bot_send_message(self) -> None:
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        delivery = TelegramDelivery(bot=mock_bot)

        await delivery.send_text(chat_id=123, text="Привет!")

        mock_bot.send_message.assert_called_once_with(chat_id=123, text="Привет!")

    @pytest.mark.asyncio
    async def test_send_photo_calls_bot_send_photo(self) -> None:
        mock_bot = MagicMock()
        mock_bot.send_photo = AsyncMock()
        delivery = TelegramDelivery(bot=mock_bot)

        await delivery.send_photo(chat_id=123, photo_bytes=b"fake_png", caption="Котик")

        mock_bot.send_photo.assert_called_once()
        call_kwargs = mock_bot.send_photo.call_args.kwargs
        assert call_kwargs["chat_id"] == 123
        assert call_kwargs["caption"] == "Котик"
