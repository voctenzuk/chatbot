"""Tests for proactive messaging scheduler."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestProactiveSchedulerCore:
    """Tests for ProactiveScheduler logic (anti-spam, quiet hours, sending)."""

    @pytest.fixture
    def mock_bot(self):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        return bot

    @pytest.fixture
    def scheduler(self, mock_bot):
        with patch("bot.services.proactive_scheduler.AsyncIOScheduler"):
            from bot.services.proactive_scheduler import ProactiveScheduler

            return ProactiveScheduler(mock_bot)

    @pytest.mark.asyncio
    async def test_send_proactive_message_happy_path(self, scheduler, mock_bot):
        """Message is generated via LLM and sent via bot.send_message."""
        from bot.services.llm_service import LLMResponse

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="Доброе утро! ☀️", model="test", tokens_in=10, tokens_out=20
            )
        )

        with (
            patch("bot.services.proactive_scheduler.get_llm_service", return_value=mock_llm),
            patch("bot.services.proactive_scheduler.get_system_prompt", return_value="test prompt"),
            patch.object(scheduler, "_is_quiet_hours", return_value=False),
            patch("bot.services.episode_manager.get_episode_manager") as mock_em,
        ):
            mock_manager = MagicMock()
            mock_manager.db = None
            mock_em.return_value = mock_manager

            result = await scheduler.send_proactive_message(123, "утреннее приветствие")

        assert result is True
        mock_bot.send_message.assert_called_once_with(chat_id=123, text="Доброе утро! ☀️")
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_skipped_during_quiet_hours(self, scheduler, mock_bot):
        """No message sent during quiet hours (23:00-08:00)."""
        with patch.object(scheduler, "_is_quiet_hours", return_value=True):
            result = await scheduler.send_proactive_message(123, "test")

        assert result is False
        mock_bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_anti_spam_blocks_after_max(self, scheduler, mock_bot):
        """After 3 messages/day, further sends are blocked."""
        from bot.services.llm_service import LLMResponse

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(content="hey", model="test", tokens_in=5, tokens_out=10)
        )

        with (
            patch("bot.services.proactive_scheduler.get_llm_service", return_value=mock_llm),
            patch("bot.services.proactive_scheduler.get_system_prompt", return_value="test"),
            patch.object(scheduler, "_is_quiet_hours", return_value=False),
            patch("bot.services.episode_manager.get_episode_manager") as mock_em,
        ):
            mock_manager = MagicMock()
            mock_manager.db = None
            mock_em.return_value = mock_manager

            for _ in range(3):
                result = await scheduler.send_proactive_message(123, "test")
                assert result is True

            result = await scheduler.send_proactive_message(123, "test")
            assert result is False

        assert mock_bot.send_message.call_count == 3

    @pytest.mark.asyncio
    async def test_send_llm_error_swallowed(self, scheduler, mock_bot):
        """LLM error is caught and logged, returns False."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM timeout"))

        with (
            patch("bot.services.proactive_scheduler.get_llm_service", return_value=mock_llm),
            patch("bot.services.proactive_scheduler.get_system_prompt", return_value="test"),
            patch.object(scheduler, "_is_quiet_hours", return_value=False),
        ):
            result = await scheduler.send_proactive_message(123, "test")

        assert result is False
        mock_bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_telegram_error_swallowed(self, scheduler, mock_bot):
        """bot.send_message error is caught and logged, returns False."""
        from bot.services.llm_service import LLMResponse

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(content="hi", model="test", tokens_in=5, tokens_out=10)
        )
        mock_bot.send_message = AsyncMock(side_effect=Exception("Forbidden: bot was blocked"))

        with (
            patch("bot.services.proactive_scheduler.get_llm_service", return_value=mock_llm),
            patch("bot.services.proactive_scheduler.get_system_prompt", return_value="test"),
            patch.object(scheduler, "_is_quiet_hours", return_value=False),
        ):
            result = await scheduler.send_proactive_message(123, "test")

        assert result is False

    def test_quiet_hours_at_2am(self, scheduler):
        """2:00 AM is quiet hours."""
        with patch("bot.services.proactive_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 19, 2, 0)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            assert scheduler._is_quiet_hours() is True

    def test_quiet_hours_at_10am(self, scheduler):
        """10:00 AM is not quiet hours."""
        with patch("bot.services.proactive_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 19, 10, 0)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            assert scheduler._is_quiet_hours() is False


class TestProactiveSchedulerTriggers:
    """Tests for scheduler trigger methods."""

    @pytest.fixture
    def mock_bot(self):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        return bot

    @pytest.fixture
    def scheduler(self, mock_bot):
        with patch("bot.services.proactive_scheduler.AsyncIOScheduler"):
            from bot.services.proactive_scheduler import ProactiveScheduler

            return ProactiveScheduler(mock_bot)

    @pytest.mark.asyncio
    async def test_morning_check_sends_to_all_users(self, scheduler):
        """Morning check sends to all active users."""
        scheduler._get_active_user_ids = AsyncMock(return_value=[111, 222])
        scheduler.send_proactive_message = AsyncMock(return_value=True)

        await scheduler._morning_check()

        assert scheduler.send_proactive_message.call_count == 2
        scheduler.send_proactive_message.assert_any_call(111, "утреннее приветствие")
        scheduler.send_proactive_message.assert_any_call(222, "утреннее приветствие")

    @pytest.mark.asyncio
    async def test_idle_check_only_sends_to_idle_users(self, scheduler):
        """Idle check only messages users silent for >6 hours."""
        scheduler._get_active_user_ids = AsyncMock(return_value=[111, 222])

        now = datetime.now()

        async def mock_last_msg(uid):
            if uid == 111:
                return now - timedelta(hours=8)
            return now - timedelta(hours=1)

        scheduler._get_last_message_time = AsyncMock(side_effect=mock_last_msg)
        scheduler.send_proactive_message = AsyncMock(return_value=True)

        await scheduler._idle_check()

        scheduler.send_proactive_message.assert_called_once_with(111, "давно не общались, скучаю")

    @pytest.mark.asyncio
    async def test_idle_check_no_active_users(self, scheduler):
        """Idle check with no active users completes silently."""
        scheduler._get_active_user_ids = AsyncMock(return_value=[])
        scheduler.send_proactive_message = AsyncMock()

        await scheduler._idle_check()

        scheduler.send_proactive_message.assert_not_called()
