"""Proactive messaging scheduler — bot initiates conversations.

Uses APScheduler with optional Redis job store for persistent scheduling.
Triggers: morning greeting (09:00), evening check-in (21:00),
idle detection (user silent >6h). Anti-spam: max 3 messages/day/user,
quiet hours 23:00-08:00.

Architecture:
    app.py startup
    ┌──────────────────┐
    │ AsyncIOScheduler │──→ Redis job store (or memory)
    │ .start()         │
    └──────┬───────────┘
           ├── CronTrigger(hour=9)   → _morning_check()
           ├── CronTrigger(hour=21)  → _evening_check()
           └── IntervalTrigger(30m)  → _idle_check()
                        │
                        ▼
             Anti-spam → LLM generate → delivery.send_text
"""

import importlib.util
from datetime import datetime, timedelta

from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from bot.config import settings
from bot.conversation.system_prompt import get_system_prompt
from bot.llm.service import get_llm_service
from bot.ports import MessageDeliveryPort

REDIS_JOBSTORE_AVAILABLE = importlib.util.find_spec("apscheduler.jobstores.redis") is not None

_QUIET_HOUR_START = 23
_QUIET_HOUR_END = 8
_MAX_PROACTIVE_PER_DAY = 3
_IDLE_THRESHOLD_HOURS = 6


class ProactiveScheduler:
    """Schedules and sends proactive messages to users.

    Uses APScheduler for cron/interval triggers with optional Redis
    persistence. Falls back to in-memory job store if Redis unavailable.
    """

    def __init__(self, delivery: MessageDeliveryPort) -> None:
        self._delivery = delivery
        self._send_counts: dict[int, dict[str, int]] = {}

        jobstores: dict[str, MemoryJobStore] = {}
        if REDIS_JOBSTORE_AVAILABLE and settings.redis_url:
            try:
                from apscheduler.jobstores.redis import RedisJobStore as _RedisJobStore

                jobstores["default"] = _RedisJobStore(url=settings.redis_url)
                logger.info("ProactiveScheduler using Redis job store")
            except Exception as e:
                logger.warning("Redis job store failed, using memory: {}", e)
                jobstores["default"] = MemoryJobStore()
        else:
            jobstores["default"] = MemoryJobStore()

        self._scheduler = AsyncIOScheduler(jobstores=jobstores)

    def start(self) -> None:
        """Start the scheduler with all triggers."""
        self._scheduler.add_job(
            self._morning_check,
            "cron",
            hour=9,
            minute=0,
            id="proactive_morning",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._evening_check,
            "cron",
            hour=21,
            minute=0,
            id="proactive_evening",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._idle_check,
            "interval",
            minutes=30,
            id="proactive_idle",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info("ProactiveScheduler started with 3 triggers")

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("ProactiveScheduler stopped")

    def _is_quiet_hours(self) -> bool:
        """Check if current time is in quiet hours (23:00-08:00)."""
        hour = datetime.now().hour
        return hour >= _QUIET_HOUR_START or hour < _QUIET_HOUR_END

    def _check_anti_spam(self, user_id: int) -> bool:
        """Return True if user can receive more proactive messages today."""
        today = datetime.now().strftime("%Y-%m-%d")
        user_counts = self._send_counts.get(user_id, {})
        return user_counts.get(today, 0) < _MAX_PROACTIVE_PER_DAY

    def _record_send(self, user_id: int) -> None:
        """Record that a proactive message was sent to user."""
        today = datetime.now().strftime("%Y-%m-%d")
        if user_id not in self._send_counts:
            self._send_counts[user_id] = {}
        self._send_counts[user_id][today] = self._send_counts[user_id].get(today, 0) + 1
        self._send_counts[user_id] = {
            k: v for k, v in self._send_counts[user_id].items() if k == today
        }

    async def _get_active_user_ids(self) -> list[int]:
        """Get list of active user IDs from the database."""
        try:
            from bot.infra.db_client import get_db_client

            db = get_db_client()
            return await db.get_all_user_ids()
        except Exception as exc:
            logger.warning("Failed to get active users: {}", exc)
        return []

    async def _get_last_message_time(self, user_id: int) -> datetime | None:
        """Get the last message timestamp for a user."""
        try:
            from bot.infra.db_client import get_db_client

            db = get_db_client()
            episode = await db.get_active_episode_for_user(user_id)
            if episode and hasattr(episode, "last_user_message_at"):
                return episode.last_user_message_at
        except Exception as exc:
            logger.warning("Failed to get last message time for user {}: {}", user_id, exc)
        return None

    async def send_proactive_message(self, user_id: int, prompt_hint: str) -> bool:
        """Generate and send a proactive message to user."""
        if self._is_quiet_hours():
            logger.debug("Quiet hours — skipping proactive message for user {}", user_id)
            return False

        if not self._check_anti_spam(user_id):
            logger.debug("Anti-spam limit — skipping proactive message for user {}", user_id)
            return False

        try:
            from bot.infra.langfuse_service import get_langfuse_service

            system_prompt = get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "system",
                    "content": (
                        f"Ты решила написать первой. Причина: {prompt_hint}. "
                        "Напиши короткое (1-2 предложения) естественное сообщение. "
                        "Не начинай с 'Привет' каждый раз — будь разнообразной."
                    ),
                },
                {"role": "user", "content": "(ожидает твоё сообщение)"},
            ]

            with get_langfuse_service().trace(
                user_id=user_id,
                trace_name="proactive",
                tags=["proactive"],
            ):
                llm_response = await get_llm_service().generate(messages)
            await self._delivery.send_text(chat_id=user_id, text=llm_response.content)
            self._record_send(user_id)

            try:
                from bot.conversation.episode_manager import get_episode_manager

                manager = get_episode_manager()
                if manager.db is not None:
                    await manager.process_assistant_message(
                        user_id=user_id,
                        content=llm_response.content,
                        tokens_in=llm_response.tokens_in,
                        tokens_out=llm_response.tokens_out,
                        model=llm_response.model,
                    )
            except Exception as persist_exc:
                logger.warning(
                    "Failed to persist proactive message for user {}: {}", user_id, persist_exc
                )

            logger.info("Sent proactive message to user {} ({})", user_id, prompt_hint)
            return True

        except Exception as exc:
            logger.warning("Failed to send proactive message to user {}: {}", user_id, exc)
            return False

    async def _morning_check(self) -> None:
        """Send morning greetings to active users."""
        user_ids = await self._get_active_user_ids()
        for user_id in user_ids:
            await self.send_proactive_message(user_id, "утреннее приветствие")

    async def _evening_check(self) -> None:
        """Send evening check-ins to active users."""
        user_ids = await self._get_active_user_ids()
        for user_id in user_ids:
            await self.send_proactive_message(user_id, "вечерний вопрос — как прошёл день?")

    async def _idle_check(self) -> None:
        """Check for idle users and send 'miss you' messages."""
        user_ids = await self._get_active_user_ids()
        threshold = datetime.now() - timedelta(hours=_IDLE_THRESHOLD_HOURS)

        for user_id in user_ids:
            last_msg = await self._get_last_message_time(user_id)
            if last_msg is not None and last_msg < threshold:
                await self.send_proactive_message(user_id, "давно не общались, скучаю")


# ---------------------------------------------------------------------------
# Dependency-injection helpers
# ---------------------------------------------------------------------------

_scheduler: ProactiveScheduler | None = None


def get_proactive_scheduler() -> ProactiveScheduler | None:
    """Get the global ProactiveScheduler instance."""
    return _scheduler


def set_proactive_scheduler(scheduler: ProactiveScheduler | None) -> None:
    """Set the global ProactiveScheduler instance."""
    global _scheduler
    _scheduler = scheduler
