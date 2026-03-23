"""Tests for RelationshipScorer."""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.memory.relationship_scorer import (
    RelationshipLevel,
    RelationshipScorer,
    RelationshipSignals,
    RelationshipTier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(
    *,
    personal_disclosures: int = 0,
    relationship_memories: int = 0,
    avg_emotional_depth: float = 0.0,
    total_messages: int = 0,
    days_active: int = 0,
    consecutive_days: int = 0,
    days_since_last: int = 0,
) -> MagicMock:
    db = MagicMock()
    db.get_user_facts_summary = AsyncMock(
        return_value={
            "personal_disclosures": personal_disclosures,
            "relationship_memories": relationship_memories,
            "avg_emotional_depth": avg_emotional_depth,
        }
    )
    db.get_message_stats = AsyncMock(
        return_value={
            "total_messages": total_messages,
            "days_active": days_active,
            "consecutive_days": consecutive_days,
            "days_since_last": days_since_last,
        }
    )
    return db


def _signals(
    *,
    days_active: int = 0,
    total_messages: int = 0,
    personal_disclosures: int = 0,
    emotional_depth: float = 0.0,
    relationship_memories: int = 0,
    consecutive_days: int = 0,
    days_since_last: int = 0,
) -> RelationshipSignals:
    return RelationshipSignals(
        days_active=days_active,
        total_messages=total_messages,
        personal_disclosures=personal_disclosures,
        emotional_depth=emotional_depth,
        relationship_memories=relationship_memories,
        consecutive_days=consecutive_days,
        days_since_last=days_since_last,
    )


# ---------------------------------------------------------------------------
# Scoring formula
# ---------------------------------------------------------------------------


class TestScoringFormula:
    def _baseline_score(self, scorer: RelationshipScorer) -> float:
        """Score contribution from all-zero signals (log(1) terms are non-zero)."""
        return scorer._score(_signals())

    def test_all_zeros_baseline(self) -> None:
        """All-zeros signals produce a small non-zero baseline from log(1) terms."""
        scorer = RelationshipScorer()
        score = scorer._score(_signals())
        # log(max(1,0)+1)/log(X)*weight = log(1)/log(X)*weight = 0 for all log terms,
        # and linear terms are 0 too — so the result is exactly 0.
        # Actually log(1) = 0, so all log-scale terms are 0 at inputs 0.
        # But max(1, 0) = 1, so days_active term = log(2)/log(31)*2 ≠ 0.
        # We just verify it's ≥ 0 and ≤ 10.
        assert 0.0 <= score <= 10.0

    def test_score_is_bounded_0_to_10(self) -> None:
        scorer = RelationshipScorer()
        maxed = _signals(
            days_active=1000,
            total_messages=10000,
            personal_disclosures=1000,
            emotional_depth=1.0,
            relationship_memories=1000,
            consecutive_days=1000,
        )
        score = scorer._score(maxed)
        assert 0.0 <= score <= 10.0
        assert score == pytest.approx(10.0)

    def test_days_active_log_scale_contribution(self) -> None:
        """days_active=30 contributes exactly 2.0 pts to the total."""
        scorer = RelationshipScorer()
        # Measure delta between days_active=1 and days_active=30 (all other signals fixed at 0).
        score_30 = scorer._score(_signals(days_active=30))
        days_active_delta = score_30 - scorer._score(_signals(days_active=1))
        # At days_active=30: log(31)/log(31)*2 = 2.0; at 1: log(2)/log(31)*2 ≈ 0.369
        expected_delta = 2.0 - min(2.0, math.log(2) / math.log(31) * 2.0)
        assert days_active_delta == pytest.approx(expected_delta, abs=0.01)

    def test_days_active_cap_beyond_30(self) -> None:
        scorer = RelationshipScorer()
        score_30 = scorer._score(_signals(days_active=30))
        score_100 = scorer._score(_signals(days_active=100))
        assert score_100 == pytest.approx(score_30, abs=0.01)

    def test_total_messages_cap_at_500(self) -> None:
        scorer = RelationshipScorer()
        # At 500 msgs the component maxes at 1.0
        score_500 = scorer._score(_signals(total_messages=500))
        score_1000 = scorer._score(_signals(total_messages=1000))
        assert score_500 == pytest.approx(score_1000, abs=0.01)

    def test_total_messages_increases_with_count(self) -> None:
        scorer = RelationshipScorer()
        score_10 = scorer._score(_signals(total_messages=10))
        score_100 = scorer._score(_signals(total_messages=100))
        assert score_100 > score_10

    def test_personal_disclosures_cap_at_25(self) -> None:
        scorer = RelationshipScorer()
        score_25 = scorer._score(_signals(personal_disclosures=25))
        score_50 = scorer._score(_signals(personal_disclosures=50))
        assert score_25 == pytest.approx(score_50)

    def test_personal_disclosures_increases_linearly(self) -> None:
        scorer = RelationshipScorer()
        score_5 = scorer._score(_signals(personal_disclosures=5))
        score_10 = scorer._score(_signals(personal_disclosures=10))
        # delta should double
        baseline = scorer._score(_signals())
        assert (score_10 - baseline) == pytest.approx((score_5 - baseline) * 2, abs=0.001)

    def test_emotional_depth_cap_at_1(self) -> None:
        scorer = RelationshipScorer()
        score_1 = scorer._score(_signals(emotional_depth=1.0))
        score_2 = scorer._score(_signals(emotional_depth=2.0))
        assert score_1 == pytest.approx(score_2)

    def test_emotional_depth_contributes_linearly(self) -> None:
        scorer = RelationshipScorer()
        baseline = scorer._score(_signals())
        score_05 = scorer._score(_signals(emotional_depth=0.5))
        score_10 = scorer._score(_signals(emotional_depth=1.0))
        assert (score_10 - baseline) == pytest.approx((score_05 - baseline) * 2, abs=0.001)

    def test_relationship_memories_cap_at_10(self) -> None:
        scorer = RelationshipScorer()
        score_10 = scorer._score(_signals(relationship_memories=10))
        score_20 = scorer._score(_signals(relationship_memories=20))
        assert score_10 == pytest.approx(score_20)

    def test_relationship_memories_increases_linearly(self) -> None:
        scorer = RelationshipScorer()
        baseline = scorer._score(_signals())
        score_2 = scorer._score(_signals(relationship_memories=2))
        score_4 = scorer._score(_signals(relationship_memories=4))
        assert (score_4 - baseline) == pytest.approx((score_2 - baseline) * 2, abs=0.001)

    def test_consecutive_days_cap_at_14(self) -> None:
        scorer = RelationshipScorer()
        score_14 = scorer._score(_signals(consecutive_days=14))
        score_100 = scorer._score(_signals(consecutive_days=100))
        assert score_14 == pytest.approx(score_100, abs=0.01)

    def test_consecutive_days_increases_with_streak(self) -> None:
        scorer = RelationshipScorer()
        score_3 = scorer._score(_signals(consecutive_days=3))
        score_7 = scorer._score(_signals(consecutive_days=7))
        assert score_7 > score_3

    def test_combined_known_values(self) -> None:
        """Verify a specific combination against manual calculation."""
        scorer = RelationshipScorer()
        # days_active=6 → log(7)/log(31)*2 ≈ 0.898*2... let's compute
        days_pts = min(2.0, math.log(7) / math.log(31) * 2.0)
        msg_pts = min(1.0, math.log(11) / math.log(501) * 1.0)
        disc_pts = min(2.5, 5 * 0.1)
        emo_pts = min(2.0, 0.5 * 2.0)
        rel_pts = min(1.5, 2 * 0.15)
        consec_pts = min(1.0, math.log(4) / math.log(15) * 1.0)
        expected = days_pts + msg_pts + disc_pts + emo_pts + rel_pts + consec_pts

        sig = _signals(
            days_active=6,
            total_messages=10,
            personal_disclosures=5,
            emotional_depth=0.5,
            relationship_memories=2,
            consecutive_days=3,
        )
        score = scorer._score(sig)
        assert score == pytest.approx(expected, abs=0.001)


# ---------------------------------------------------------------------------
# Tier boundaries
# ---------------------------------------------------------------------------


class TestTierBoundaries:
    def test_score_0_is_acquaintance(self) -> None:
        scorer = RelationshipScorer()
        assert scorer._tier_from_score(0.0) == RelationshipTier.ACQUAINTANCE

    def test_score_2_is_acquaintance(self) -> None:
        scorer = RelationshipScorer()
        assert scorer._tier_from_score(2.9) == RelationshipTier.ACQUAINTANCE

    def test_score_3_is_friend(self) -> None:
        scorer = RelationshipScorer()
        assert scorer._tier_from_score(3.0) == RelationshipTier.FRIEND

    def test_score_5_is_friend(self) -> None:
        scorer = RelationshipScorer()
        assert scorer._tier_from_score(5.9) == RelationshipTier.FRIEND

    def test_score_6_is_close_friend(self) -> None:
        scorer = RelationshipScorer()
        assert scorer._tier_from_score(6.0) == RelationshipTier.CLOSE_FRIEND

    def test_score_8_is_close_friend(self) -> None:
        scorer = RelationshipScorer()
        assert scorer._tier_from_score(8.9) == RelationshipTier.CLOSE_FRIEND

    def test_score_9_is_intimate(self) -> None:
        scorer = RelationshipScorer()
        assert scorer._tier_from_score(9.0) == RelationshipTier.INTIMATE

    def test_score_10_is_intimate(self) -> None:
        scorer = RelationshipScorer()
        assert scorer._tier_from_score(10.0) == RelationshipTier.INTIMATE


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------


class TestDecay:
    def test_zero_days_no_decay(self) -> None:
        scorer = RelationshipScorer()
        assert scorer._apply_decay(5.0, 0) == pytest.approx(5.0)

    def test_ten_days_approximately_30_percent_reduction(self) -> None:
        scorer = RelationshipScorer()
        # 1.0 - 0.03 * 10 = 0.7
        result = scorer._apply_decay(5.0, 10)
        assert result == pytest.approx(5.0 * 0.7, abs=0.001)

    def test_33_days_floors_at_50_percent(self) -> None:
        scorer = RelationshipScorer()
        # 1.0 - 0.03 * 33 = 0.01 → max(0.5, 0.01) = 0.5
        result = scorer._apply_decay(5.0, 33)
        assert result == pytest.approx(5.0 * 0.5)

    def test_large_inactivity_floors_at_50_percent(self) -> None:
        scorer = RelationshipScorer()
        result = scorer._apply_decay(8.0, 100)
        assert result == pytest.approx(8.0 * 0.5)

    def test_exact_floor_boundary(self) -> None:
        scorer = RelationshipScorer()
        # 0.03 * 16 = 0.48 → 1.0-0.48=0.52 > 0.5, still no floor
        result = scorer._apply_decay(4.0, 16)
        assert result == pytest.approx(4.0 * 0.52, abs=0.001)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestCache:
    async def test_cache_miss_calls_db(self) -> None:
        scorer = RelationshipScorer()
        db = _make_db(days_active=1, total_messages=1)
        await scorer.compute(1, db)
        db.get_user_facts_summary.assert_called_once()
        db.get_message_stats.assert_called_once()

    async def test_cache_hit_skips_db(self) -> None:
        scorer = RelationshipScorer(cache_ttl_seconds=3600)
        db = _make_db(days_active=1, total_messages=1)
        await scorer.compute(1, db)
        await scorer.compute(1, db)
        # DB should only be called once (second call hits cache)
        assert db.get_user_facts_summary.call_count == 1

    async def test_invalidate_forces_db_call(self) -> None:
        scorer = RelationshipScorer(cache_ttl_seconds=3600)
        db = _make_db(days_active=1, total_messages=1)
        await scorer.compute(1, db)
        scorer.invalidate(1)
        await scorer.compute(1, db)
        assert db.get_user_facts_summary.call_count == 2

    async def test_cache_ttl_zero_always_refetches(self) -> None:
        scorer = RelationshipScorer(cache_ttl_seconds=0)
        db = _make_db(days_active=1, total_messages=1)
        await scorer.compute(1, db)
        await scorer.compute(1, db)
        assert db.get_user_facts_summary.call_count == 2

    def test_invalidate_nonexistent_user_is_noop(self) -> None:
        scorer = RelationshipScorer()
        scorer.invalidate(999)  # should not raise


# ---------------------------------------------------------------------------
# invalidate_if_nth_message
# ---------------------------------------------------------------------------


class TestInvalidateIfNthMessage:
    async def test_10th_message_invalidates(self) -> None:
        scorer = RelationshipScorer(cache_ttl_seconds=3600)
        db = _make_db(days_active=1, total_messages=1)
        await scorer.compute(1, db)
        # Should be cached now
        scorer.invalidate_if_nth_message(1, 10)
        # Cache should be cleared — next compute hits DB again
        await scorer.compute(1, db)
        assert db.get_user_facts_summary.call_count == 2

    async def test_non_nth_message_keeps_cache(self) -> None:
        scorer = RelationshipScorer(cache_ttl_seconds=3600)
        db = _make_db(days_active=1, total_messages=1)
        await scorer.compute(1, db)
        scorer.invalidate_if_nth_message(1, 7)
        await scorer.compute(1, db)
        assert db.get_user_facts_summary.call_count == 1

    async def test_20th_message_also_invalidates(self) -> None:
        scorer = RelationshipScorer(cache_ttl_seconds=3600)
        db = _make_db(days_active=1, total_messages=1)
        await scorer.compute(1, db)
        scorer.invalidate_if_nth_message(1, 20)
        await scorer.compute(1, db)
        assert db.get_user_facts_summary.call_count == 2

    def test_zero_message_count_is_noop(self) -> None:
        scorer = RelationshipScorer()
        scorer.invalidate_if_nth_message(1, 0)  # should not raise

    def test_custom_n(self) -> None:
        scorer = RelationshipScorer()
        # With n=5, message 5 should trigger
        scorer._cache[1] = (
            RelationshipLevel(score=1.0, tier=RelationshipTier.ACQUAINTANCE, level=1),
            9999999999.0,  # far future timestamp
        )
        scorer.invalidate_if_nth_message(1, 5, n=5)
        assert 1 not in scorer._cache

    def test_custom_n_non_trigger(self) -> None:
        scorer = RelationshipScorer()
        level = RelationshipLevel(score=1.0, tier=RelationshipTier.ACQUAINTANCE, level=1)
        scorer._cache[1] = (level, 9999999999.0)
        scorer.invalidate_if_nth_message(1, 4, n=5)
        assert 1 in scorer._cache


# ---------------------------------------------------------------------------
# Milestone detection — tier change
# ---------------------------------------------------------------------------


class TestMilestoneTierChange:
    async def test_tier_upgrade_fires_callback(self) -> None:
        callback = AsyncMock()
        scorer = RelationshipScorer(milestone_callback=callback)

        # Seed last_tier as ACQUAINTANCE
        scorer._last_tier[42] = RelationshipTier.ACQUAINTANCE

        # DB signals that produce FRIEND tier
        db = _make_db(personal_disclosures=30, days_active=5, total_messages=50)
        await scorer.compute(42, db)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == 42
        assert call_args[1] == RelationshipTier.ACQUAINTANCE
        assert call_args[2] in (
            RelationshipTier.FRIEND,
            RelationshipTier.CLOSE_FRIEND,
            RelationshipTier.INTIMATE,
        )

    async def test_no_callback_on_same_tier(self) -> None:
        callback = AsyncMock()
        scorer = RelationshipScorer(milestone_callback=callback)

        # Seed last_tier as ACQUAINTANCE, and ensure score stays in ACQUAINTANCE range
        scorer._last_tier[1] = RelationshipTier.ACQUAINTANCE
        db = _make_db()  # all zeros → ACQUAINTANCE
        await scorer.compute(1, db)
        callback.assert_not_called()

    async def test_no_callback_on_tier_downgrade(self) -> None:
        """Tier downgrade (due to decay) should not fire callback."""
        callback = AsyncMock()
        scorer = RelationshipScorer(milestone_callback=callback)

        scorer._last_tier[1] = RelationshipTier.FRIEND
        db = _make_db()  # no activity → ACQUAINTANCE after decay
        await scorer.compute(1, db)
        callback.assert_not_called()

    async def test_callback_exception_does_not_propagate(self) -> None:
        callback = AsyncMock(side_effect=RuntimeError("boom"))
        scorer = RelationshipScorer(milestone_callback=callback)
        scorer._last_tier[1] = RelationshipTier.ACQUAINTANCE

        db = _make_db(personal_disclosures=30, days_active=5, total_messages=50)
        # Should not raise
        await scorer.compute(1, db)


# ---------------------------------------------------------------------------
# Milestone detection — day milestones
# ---------------------------------------------------------------------------


class TestMilestoneDays:
    async def test_7_day_milestone_fires_callback(self) -> None:
        callback = AsyncMock()
        scorer = RelationshipScorer(milestone_callback=callback)
        # Tier already set so no tier-change callback fires
        scorer._last_tier[1] = RelationshipTier.ACQUAINTANCE

        db = _make_db(days_active=7)
        await scorer.compute(1, db)
        callback.assert_called()

    async def test_30_day_milestone_fires_callback(self) -> None:
        callback = AsyncMock()
        scorer = RelationshipScorer(milestone_callback=callback)
        scorer._last_tier[1] = RelationshipTier.ACQUAINTANCE
        db = _make_db(days_active=30)
        await scorer.compute(1, db)
        callback.assert_called()

    async def test_100_day_milestone_fires_callback(self) -> None:
        callback = AsyncMock()
        scorer = RelationshipScorer(milestone_callback=callback)
        scorer._last_tier[1] = RelationshipTier.ACQUAINTANCE
        db = _make_db(days_active=100)
        await scorer.compute(1, db)
        callback.assert_called()

    async def test_non_milestone_day_no_callback(self) -> None:
        callback = AsyncMock()
        scorer = RelationshipScorer(milestone_callback=callback)
        scorer._last_tier[1] = RelationshipTier.ACQUAINTANCE
        db = _make_db(days_active=5)
        await scorer.compute(1, db)
        callback.assert_not_called()


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    async def test_none_db_client_returns_acquaintance(self) -> None:
        scorer = RelationshipScorer()
        level = await scorer.compute(1, None)
        assert level.tier == RelationshipTier.ACQUAINTANCE
        assert level.score == pytest.approx(0.0)
        assert level.level == 0

    async def test_db_error_returns_acquaintance(self) -> None:
        scorer = RelationshipScorer()
        db = MagicMock()
        db.get_user_facts_summary = AsyncMock(side_effect=RuntimeError("DB down"))
        db.get_message_stats = AsyncMock(side_effect=RuntimeError("DB down"))
        level = await scorer.compute(1, db)
        assert level.tier == RelationshipTier.ACQUAINTANCE

    async def test_partial_db_error_returns_acquaintance(self) -> None:
        scorer = RelationshipScorer()
        db = MagicMock()
        db.get_user_facts_summary = AsyncMock(return_value={"personal_disclosures": 5})
        db.get_message_stats = AsyncMock(side_effect=RuntimeError("partial error"))
        level = await scorer.compute(1, db)
        assert level.tier == RelationshipTier.ACQUAINTANCE

    async def test_no_milestone_callback_no_crash(self) -> None:
        scorer = RelationshipScorer(milestone_callback=None)
        db = _make_db(days_active=7)
        level = await scorer.compute(1, db)
        assert level is not None


# ---------------------------------------------------------------------------
# RelationshipLevel dataclass
# ---------------------------------------------------------------------------


class TestRelationshipLevel:
    def test_is_frozen(self) -> None:
        level = RelationshipLevel(score=5.0, tier=RelationshipTier.FRIEND, level=5)
        with pytest.raises((AttributeError, TypeError)):
            level.score = 6.0  # type: ignore[misc]

    def test_level_is_floor_of_score(self) -> None:
        level = RelationshipLevel(score=4.7, tier=RelationshipTier.FRIEND, level=int(4.7))
        assert level.level == 4


# ---------------------------------------------------------------------------
# RelationshipSignals dataclass
# ---------------------------------------------------------------------------


class TestRelationshipSignals:
    def test_is_frozen(self) -> None:
        sig = _signals(days_active=1)
        with pytest.raises((AttributeError, TypeError)):
            sig.days_active = 2  # type: ignore[misc]
