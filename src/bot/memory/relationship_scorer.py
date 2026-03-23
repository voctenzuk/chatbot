"""Relationship scorer — computes relationship level from structured fact signals."""

import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from loguru import logger

_DAY_MILESTONES = frozenset({7, 30, 100})


class RelationshipTier(StrEnum):
    ACQUAINTANCE = "acquaintance"  # 0-2
    FRIEND = "friend"  # 3-5
    CLOSE_FRIEND = "close_friend"  # 6-8
    INTIMATE = "intimate"  # 9-10


@dataclass(frozen=True)
class RelationshipSignals:
    """Raw signals for relationship level computation."""

    days_active: int
    total_messages: int
    personal_disclosures: int  # category IN ('semantic', 'emotional')
    emotional_depth: float  # avg abs(emotional_valence)
    relationship_memories: int  # memory_type IN ('milestone','inside_joke','boundary')
    consecutive_days: int
    days_since_last: int  # for decay


@dataclass(frozen=True)
class RelationshipLevel:
    """Computed relationship level with tier label."""

    score: float  # 0.0-10.0
    tier: RelationshipTier
    level: int  # floor of score


_FALLBACK_LEVEL = RelationshipLevel(score=0.0, tier=RelationshipTier.ACQUAINTANCE, level=0)


class RelationshipScorer:
    """Compute relationship level from structured fact signals.

    Scoring formula (max 10.0):
    - days_active:           0-2.0 pts (log scale, cap at 30 days)
    - total_messages:        0-1.0 pts (log scale, cap at 500)
    - personal_disclosures:  0-2.5 pts (linear, cap at 25 facts)
    - emotional_depth:       0-2.0 pts (avg abs(valence) * 2)
    - relationship_memories: 0-1.5 pts (linear, cap at 10)
    - consecutive_days:      0-1.0 pts (log scale, cap at 14)

    Cache TTL defaults to 1 hour. Invalidated on new fact upsert.
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 3600,
        milestone_callback: Callable[[int, RelationshipTier, RelationshipTier], Awaitable[None]]
        | None = None,
    ) -> None:
        self._cache_ttl = cache_ttl_seconds
        self._milestone_callback = milestone_callback
        # user_id → (level, timestamp)
        self._cache: dict[int, tuple[RelationshipLevel, float]] = {}
        # track last known tier for milestone detection
        self._last_tier: dict[int, RelationshipTier] = {}
        # track fired day milestones to avoid repeated callbacks
        self._fired_day_milestones: dict[int, set[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compute(self, user_id: int, db_client: Any) -> RelationshipLevel:
        """Compute relationship level, using cache when fresh.

        Steps:
        1. Return cached entry if within TTL.
        2. Fetch signals from DB (two RPCs).
        3. Score signals, apply decay.
        4. Detect tier milestone and fire callback if tier rose.
        5. Cache and return result.

        Gracefully degrades to ACQUAINTANCE on any DB/RPC failure.
        """
        if db_client is None:
            return _FALLBACK_LEVEL

        # 1. Cache check
        cached = self._cache.get(user_id)
        if cached is not None:
            level, ts = cached
            if time.monotonic() - ts < self._cache_ttl:
                return level

        # 2. Fetch from DB
        try:
            facts_summary = await db_client.get_user_facts_summary(user_id)
            message_stats = await db_client.get_message_stats(user_id)
        except Exception as exc:
            logger.warning("relationship_scorer db_error user_id={} error={}", user_id, exc)
            return _FALLBACK_LEVEL

        # 3. Score
        try:
            signals = self._gather_signals(facts_summary, message_stats)
            raw_score = self._score(signals)
            score = self._apply_decay(raw_score, signals.days_since_last)
            tier = self._tier_from_score(score)
            level = RelationshipLevel(score=score, tier=tier, level=int(score))
        except Exception as exc:
            logger.warning("relationship_scorer compute_error user_id={} error={}", user_id, exc)
            return _FALLBACK_LEVEL

        logger.info(
            "relationship_computed user_id={} score={:.1f} tier={} days_active={}",
            user_id,
            score,
            tier.value,
            signals.days_active,
        )

        # 4. Milestone detection
        await self._check_milestones(user_id, tier, signals.days_active)

        # 5. Cache
        self._cache[user_id] = (level, time.monotonic())
        return level

    def set_milestone_callback(
        self,
        callback: Callable[[int, RelationshipTier, RelationshipTier], Awaitable[None]] | None,
    ) -> None:
        """Set or replace the milestone callback after construction.

        Used to break the circular dependency between RelationshipScorer and
        ProactiveScheduler in the wiring module.
        """
        self._milestone_callback = callback

    def invalidate(self, user_id: int) -> None:
        """Remove cached level for user. Called after new facts are written."""
        self._cache.pop(user_id, None)

    def invalidate_if_nth_message(self, user_id: int, message_count: int, n: int = 10) -> None:
        """Invalidate cache every Nth message for the user."""
        if message_count > 0 and message_count % n == 0:
            self.invalidate(user_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_signals(
        self, facts_summary: dict[str, Any], message_stats: dict[str, Any]
    ) -> RelationshipSignals:
        """Pure computation — build RelationshipSignals from pre-fetched DB data."""
        return RelationshipSignals(
            days_active=int(message_stats.get("days_active", 0)),
            total_messages=int(message_stats.get("total_messages", 0)),
            personal_disclosures=int(facts_summary.get("personal_disclosures", 0)),
            emotional_depth=float(facts_summary.get("avg_emotional_depth", 0.0)),
            relationship_memories=int(facts_summary.get("relationship_memories", 0)),
            consecutive_days=int(message_stats.get("consecutive_days", 0)),
            days_since_last=int(message_stats.get("days_since_last", 0)),
        )

    def _score(self, signals: RelationshipSignals) -> float:
        """Apply weighted scoring formula. Returns 0.0-10.0. Pure computation."""
        days_active_pts = min(
            2.0,
            math.log(max(1, signals.days_active) + 1) / math.log(31) * 2.0,
        )
        total_messages_pts = min(
            1.0,
            math.log(max(1, signals.total_messages) + 1) / math.log(501) * 1.0,
        )
        personal_disclosures_pts = min(2.5, signals.personal_disclosures * 0.1)
        emotional_depth_pts = min(2.0, signals.emotional_depth * 2.0)
        relationship_memories_pts = min(1.5, signals.relationship_memories * 0.15)
        consecutive_days_pts = min(
            1.0,
            math.log(max(1, signals.consecutive_days) + 1) / math.log(15) * 1.0,
        )

        total = (
            days_active_pts
            + total_messages_pts
            + personal_disclosures_pts
            + emotional_depth_pts
            + relationship_memories_pts
            + consecutive_days_pts
        )
        return min(10.0, max(0.0, total))

    def _apply_decay(self, score: float, days_since_last: int) -> float:
        """Apply inactivity decay: 3% per day, floor at 50%."""
        return score * max(0.5, 1.0 - 0.03 * days_since_last)

    def _tier_from_score(self, score: float) -> RelationshipTier:
        """Map continuous score to discrete tier."""
        if score >= 9.0:
            return RelationshipTier.INTIMATE
        if score >= 6.0:
            return RelationshipTier.CLOSE_FRIEND
        if score >= 3.0:
            return RelationshipTier.FRIEND
        return RelationshipTier.ACQUAINTANCE

    async def _check_milestones(
        self, user_id: int, new_tier: RelationshipTier, days_active: int
    ) -> None:
        """Fire milestone_callback when tier rises or day milestone hit."""
        if self._milestone_callback is None:
            return

        old_tier = self._last_tier.get(user_id)
        self._last_tier[user_id] = new_tier

        # Tier progression milestones (upward only)
        _tier_order = [
            RelationshipTier.ACQUAINTANCE,
            RelationshipTier.FRIEND,
            RelationshipTier.CLOSE_FRIEND,
            RelationshipTier.INTIMATE,
        ]
        if old_tier is not None and _tier_order.index(new_tier) > _tier_order.index(old_tier):
            try:
                await self._milestone_callback(user_id, old_tier, new_tier)
            except Exception as exc:
                logger.warning("milestone_callback failed user_id={} error={}", user_id, exc)

        # Day milestones (fire once per user per milestone day)
        if days_active in _DAY_MILESTONES:
            user_fired = self._fired_day_milestones.setdefault(user_id, set())
            if days_active not in user_fired:
                user_fired.add(days_active)
                try:
                    await self._milestone_callback(user_id, new_tier, new_tier)
                except Exception as exc:
                    logger.warning(
                        "day_milestone_callback failed user_id={} days={} error={}",
                        user_id,
                        days_active,
                        exc,
                    )
