"""Proactive messaging scheduler — bot initiates conversations.

Uses APScheduler with optional Redis job store for persistent scheduling.
Triggers: morning greeting (09:00), evening check-in (21:00),
idle detection (user silent >6h). Anti-spam: max 3 messages/day/user,
quiet hours 23:00-08:00.

Dynamic limits scale with relationship tier:
    ACQUAINTANCE → max 1/day, idle threshold 12h
    FRIEND       → max 2/day, idle threshold 8h
    CLOSE_FRIEND → max 3/day, idle threshold 6h
    INTIMATE     → max 4/day, idle threshold 4h

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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from bot.config import settings
from bot.conversation.system_prompt import get_system_prompt
from bot.infra.langfuse_service import get_langfuse_service
from bot.ports import MessageDeliveryPort

if TYPE_CHECKING:
    from apscheduler.jobstores.base import BaseJobStore

    from bot.llm.service import LLMService
    from bot.memory.relationship_scorer import RelationshipScorer, RelationshipTier

REDIS_JOBSTORE_AVAILABLE = importlib.util.find_spec("apscheduler.jobstores.redis") is not None

_QUIET_HOUR_START = 23
_QUIET_HOUR_END = 8
_DEFAULT_MAX_PROACTIVE = 3
_DEFAULT_IDLE_HOURS = 6


class ProactiveScheduler:
    """Schedules and sends proactive messages to users.

    Uses APScheduler for cron/interval triggers with optional Redis
    persistence. Falls back to in-memory job store if Redis unavailable.

    All dependencies (LLM, DB, EpisodeManager, RelationshipScorer,
    ProfileBuilder, Character) are injected via constructor — no singleton
    calls inside methods.
    """

    def __init__(
        self,
        delivery: MessageDeliveryPort,
        llm: "LLMService | None" = None,
        db_client: Any | None = None,
        episode_manager: Any | None = None,
        relationship_scorer: "RelationshipScorer | None" = None,
        profile_builder: Any | None = None,
        character: Any | None = None,
    ) -> None:
        self._delivery = delivery
        self._llm = llm
        self._db_client = db_client
        self._episode_manager = episode_manager
        self._relationship_scorer = relationship_scorer
        self._profile_builder = profile_builder
        self._character = character

        self._send_counts: dict[int, dict[str, int]] = {}
        # Separate daily counter for milestone messages (max 1/day/user)
        self._milestone_counts: dict[int, dict[str, int]] = {}

        jobstores: dict[str, BaseJobStore] = {}
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
        hour = datetime.now(tz=UTC).hour
        return hour >= _QUIET_HOUR_START or hour < _QUIET_HOUR_END

    async def _get_limits(self, user_id: int) -> tuple[int, int]:
        """Return (max_proactive_per_day, idle_threshold_hours) for user's tier.

        Falls back to defaults when RelationshipScorer or DB is unavailable.
        """
        from bot.memory.relationship_scorer import RelationshipTier

        if not self._relationship_scorer or not self._db_client:
            return (_DEFAULT_MAX_PROACTIVE, _DEFAULT_IDLE_HOURS)
        try:
            level = await self._relationship_scorer.compute(user_id, self._db_client)
            limits: dict[RelationshipTier, tuple[int, int]] = {
                RelationshipTier.ACQUAINTANCE: (1, 12),
                RelationshipTier.FRIEND: (2, 8),
                RelationshipTier.CLOSE_FRIEND: (3, 6),
                RelationshipTier.INTIMATE: (4, 4),
            }
            return limits.get(level.tier, (_DEFAULT_MAX_PROACTIVE, _DEFAULT_IDLE_HOURS))
        except Exception:
            return (_DEFAULT_MAX_PROACTIVE, _DEFAULT_IDLE_HOURS)

    async def _check_anti_spam(self, user_id: int) -> bool:
        """Return True if user can receive more proactive messages today."""
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        user_counts = self._send_counts.get(user_id, {})
        max_count, _ = await self._get_limits(user_id)
        return user_counts.get(today, 0) < max_count

    def _record_send(self, user_id: int) -> None:
        """Record that a proactive message was sent to user."""
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        if user_id not in self._send_counts:
            self._send_counts[user_id] = {}
        self._send_counts[user_id][today] = self._send_counts[user_id].get(today, 0) + 1
        self._send_counts[user_id] = {
            k: v for k, v in self._send_counts[user_id].items() if k == today
        }

    def _record_milestone_send(self, user_id: int) -> None:
        """Record that a milestone message was sent to user."""
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        if user_id not in self._milestone_counts:
            self._milestone_counts[user_id] = {}
        self._milestone_counts[user_id][today] = self._milestone_counts[user_id].get(today, 0) + 1
        self._milestone_counts[user_id] = {
            k: v for k, v in self._milestone_counts[user_id].items() if k == today
        }

    def _milestone_allowed_today(self, user_id: int) -> bool:
        """Return True if user has not yet received a milestone message today."""
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        return self._milestone_counts.get(user_id, {}).get(today, 0) < 1

    async def _get_active_user_ids(self) -> list[int]:
        """Get list of active user IDs from the database."""
        if self._db_client is None:
            return []
        try:
            return await self._db_client.get_all_user_ids()
        except Exception as exc:
            logger.warning("Failed to get active users: {}", exc)
        return []

    async def _get_last_message_time(self, user_id: int) -> datetime | None:
        """Get the last message timestamp for a user."""
        if self._db_client is None:
            return None
        try:
            episode = await self._db_client.get_active_episode_for_user(user_id)
            if episode and hasattr(episode, "last_user_message_at"):
                return episode.last_user_message_at
        except Exception as exc:
            logger.warning("Failed to get last message time for user {}: {}", user_id, exc)
        return None

    async def send_proactive_message(self, user_id: int, prompt_hint: str) -> bool:
        """Generate and send a proactive message to user.

        Returns True if message was successfully sent, False otherwise.
        """
        if self._llm is None:
            logger.debug("LLM not configured — skipping proactive message for user {}", user_id)
            return False

        if self._is_quiet_hours():
            logger.debug("Quiet hours — skipping proactive message for user {}", user_id)
            return False

        if not await self._check_anti_spam(user_id):
            logger.debug("Anti-spam limit — skipping proactive message for user {}", user_id)
            return False

        # Enrich hint with personal facts when profile is available
        enriched_hint = prompt_hint
        if self._profile_builder is not None:
            try:
                profile = await self._profile_builder.get_profile(user_id)
                if profile is not None:
                    fact_hints: list[str] = []
                    if profile.interests:
                        fact_hints.append(profile.interests[0])
                    if profile.likes:
                        fact_hints.append(profile.likes[0])
                    if fact_hints:
                        enriched_hint += (
                            f"\nВспомни что собеседнику интересно: {', '.join(fact_hints)}"
                        )
            except Exception as exc:
                logger.debug("Profile fetch failed for user {}: {}", user_id, exc)

        try:
            system_prompt = get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "system",
                    "content": (
                        f"Ты решила написать первой. Причина: {enriched_hint}. "
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
                llm_response = await self._llm.generate(messages)
            await self._delivery.send_text(chat_id=user_id, text=llm_response.content)
            self._record_send(user_id)

            if self._episode_manager is not None:
                try:
                    if getattr(self._episode_manager, "db", None) is not None:
                        await self._episode_manager.process_assistant_message(
                            user_id=user_id,
                            content=llm_response.content,
                            tokens_in=llm_response.tokens_in,
                            tokens_out=llm_response.tokens_out,
                            model=llm_response.model,
                        )
                except Exception as persist_exc:
                    logger.warning(
                        "Failed to persist proactive message for user {}: {}",
                        user_id,
                        persist_exc,
                    )

            logger.info("Sent proactive message to user {} ({})", user_id, prompt_hint)
            return True

        except Exception as exc:
            logger.warning("Failed to send proactive message to user {}: {}", user_id, exc)
            return False

    async def send_milestone_message(
        self,
        user_id: int,
        old_tier: "RelationshipTier",
        new_tier: "RelationshipTier",
    ) -> None:
        """Generate and send a milestone message when relationship tier rises.

        Bypasses normal anti-spam check — milestones are special events.
        Limited to max 1 milestone message per user per day.

        Args:
            user_id: Telegram user ID.
            old_tier: Previous relationship tier.
            new_tier: New (higher) relationship tier.
        """
        if self._llm is None:
            logger.debug("LLM not configured — skipping milestone message for user {}", user_id)
            return

        if not self._milestone_allowed_today(user_id):
            logger.debug("Milestone daily limit reached — skipping for user {}", user_id)
            return

        if self._is_quiet_hours():
            logger.debug("Quiet hours — skipping milestone message for user {}", user_id)
            return

        tier_labels_ru = {
            "acquaintance": "знакомые",
            "friend": "друзья",
            "close_friend": "близкие друзья",
            "intimate": "лучшие друзья",
        }
        old_label = tier_labels_ru.get(str(old_tier), str(old_tier))
        new_label = tier_labels_ru.get(str(new_tier), str(new_tier))
        prompt_hint = (
            f"Поздравь собеседника — ваши отношения стали теплее! "
            f"Раньше вы были {old_label}, теперь {new_label}. "
            "Расскажи об этом тепло и естественно, не называя уровни."
        )

        try:
            system_prompt = get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "system",
                    "content": (
                        f"Ты решила написать первой. Причина: {prompt_hint}. "
                        "Напиши короткое (1-2 предложения) естественное сообщение. "
                        "Будь тёплой и искренней."
                    ),
                },
                {"role": "user", "content": "(ожидает твоё сообщение)"},
            ]

            with get_langfuse_service().trace(
                user_id=user_id,
                trace_name="proactive_milestone",
                tags=["proactive", "milestone"],
            ):
                llm_response = await self._llm.generate(messages)

            await self._delivery.send_text(chat_id=user_id, text=llm_response.content)
            self._record_milestone_send(user_id)

            logger.info(
                "milestone_reached user_id={} old_tier={} new_tier={}",
                user_id,
                old_tier,
                new_tier,
            )

        except Exception as exc:
            logger.warning("Failed to send milestone message to user {}: {}", user_id, exc)

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

        for user_id in user_ids:
            _, idle_hours = await self._get_limits(user_id)
            threshold = datetime.now(tz=UTC) - timedelta(hours=idle_hours)
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
