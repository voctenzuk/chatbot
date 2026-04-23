"""Tests for Phase 4 proactive messaging enhancements and cross-memory references."""

from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_delivery():
    delivery = MagicMock()
    delivery.send_text = AsyncMock()
    delivery.send_photo = AsyncMock()
    return delivery


@pytest.fixture
def mock_llm():
    from bot.llm.service import LLMResponse

    llm = MagicMock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="Привет, как дела?", model="test", tokens_in=10, tokens_out=20
        )
    )
    return llm


@pytest.fixture
def mock_langfuse():
    lf = MagicMock()
    lf.trace = MagicMock(return_value=nullcontext())
    return lf


@pytest.fixture
def scheduler_with_deps(mock_delivery, mock_llm):
    """ProactiveScheduler fully wired with all optional deps as mocks."""
    with patch("bot.adapters.proactive_scheduler.AsyncIOScheduler"):
        from bot.adapters.proactive_scheduler import ProactiveScheduler

        mock_db = MagicMock()
        mock_episode_manager = MagicMock()
        mock_episode_manager.db = None
        mock_relationship_scorer = MagicMock()
        mock_profile_builder = MagicMock()

        return ProactiveScheduler(
            delivery=mock_delivery,
            llm=mock_llm,
            db_client=mock_db,
            episode_manager=mock_episode_manager,
            relationship_scorer=mock_relationship_scorer,
            profile_builder=mock_profile_builder,
        )


@pytest.fixture
def scheduler_no_deps(mock_delivery):
    """ProactiveScheduler with only delivery injected (all others None)."""
    with patch("bot.adapters.proactive_scheduler.AsyncIOScheduler"):
        from bot.adapters.proactive_scheduler import ProactiveScheduler

        return ProactiveScheduler(delivery=mock_delivery)


# ---------------------------------------------------------------------------
# Dynamic limits per relationship tier
# ---------------------------------------------------------------------------


class TestDynamicLimits:
    @pytest.mark.asyncio
    async def test_acquaintance_gets_1_proactive_per_day(self, scheduler_with_deps):
        """ACQUAINTANCE tier returns max 1 proactive message per day."""
        from bot.memory.relationship_scorer import RelationshipLevel, RelationshipTier

        mock_level = RelationshipLevel(score=1.0, tier=RelationshipTier.ACQUAINTANCE, level=1)
        scheduler_with_deps._relationship_scorer.compute = AsyncMock(return_value=mock_level)

        max_count, _ = await scheduler_with_deps._get_limits(user_id=42)

        assert max_count == 1

    @pytest.mark.asyncio
    async def test_intimate_gets_4_proactive_per_day(self, scheduler_with_deps):
        """INTIMATE tier returns max 4 proactive messages per day."""
        from bot.memory.relationship_scorer import RelationshipLevel, RelationshipTier

        mock_level = RelationshipLevel(score=9.5, tier=RelationshipTier.INTIMATE, level=9)
        scheduler_with_deps._relationship_scorer.compute = AsyncMock(return_value=mock_level)

        max_count, _ = await scheduler_with_deps._get_limits(user_id=42)

        assert max_count == 4

    @pytest.mark.asyncio
    async def test_idle_threshold_varies_by_tier(self, scheduler_with_deps):
        """ACQUAINTANCE threshold is 12h; INTIMATE threshold is 4h."""
        from bot.memory.relationship_scorer import RelationshipLevel, RelationshipTier

        for tier, expected_hours in [
            (RelationshipTier.ACQUAINTANCE, 12),
            (RelationshipTier.FRIEND, 8),
            (RelationshipTier.CLOSE_FRIEND, 6),
            (RelationshipTier.INTIMATE, 4),
        ]:
            mock_level = RelationshipLevel(score=5.0, tier=tier, level=5)
            scheduler_with_deps._relationship_scorer.compute = AsyncMock(return_value=mock_level)
            _, idle_hours = await scheduler_with_deps._get_limits(user_id=99)
            assert idle_hours == expected_hours, f"Expected {expected_hours}h for {tier}"

    @pytest.mark.asyncio
    async def test_fallback_when_scorer_none(self, scheduler_no_deps):
        """Returns default limits when no scorer is configured."""
        from bot.adapters.proactive_scheduler import _DEFAULT_IDLE_HOURS, _DEFAULT_MAX_PROACTIVE

        max_count, idle_hours = await scheduler_no_deps._get_limits(user_id=1)

        assert max_count == _DEFAULT_MAX_PROACTIVE
        assert idle_hours == _DEFAULT_IDLE_HOURS

    @pytest.mark.asyncio
    async def test_fallback_when_scorer_raises(self, scheduler_with_deps):
        """Returns default limits when scorer raises an exception."""
        from bot.adapters.proactive_scheduler import _DEFAULT_IDLE_HOURS, _DEFAULT_MAX_PROACTIVE

        scheduler_with_deps._relationship_scorer.compute = AsyncMock(
            side_effect=Exception("DB timeout")
        )

        max_count, idle_hours = await scheduler_with_deps._get_limits(user_id=1)

        assert max_count == _DEFAULT_MAX_PROACTIVE
        assert idle_hours == _DEFAULT_IDLE_HOURS


# ---------------------------------------------------------------------------
# Anti-spam uses dynamic limits
# ---------------------------------------------------------------------------


class TestAntiSpamAsync:
    @pytest.mark.asyncio
    async def test_check_anti_spam_async_uses_dynamic_limits(self, scheduler_with_deps):
        """_check_anti_spam uses per-user limit from _get_limits."""
        from bot.memory.relationship_scorer import RelationshipLevel, RelationshipTier

        # ACQUAINTANCE → limit 1
        mock_level = RelationshipLevel(score=1.0, tier=RelationshipTier.ACQUAINTANCE, level=1)
        scheduler_with_deps._relationship_scorer.compute = AsyncMock(return_value=mock_level)

        # First message allowed
        assert await scheduler_with_deps._check_anti_spam(77) is True

        # Record one send
        scheduler_with_deps._record_send(77)

        # Second message blocked for ACQUAINTANCE
        assert await scheduler_with_deps._check_anti_spam(77) is False

    @pytest.mark.asyncio
    async def test_intimate_allows_4_before_blocking(self, scheduler_with_deps):
        """INTIMATE tier allows 4 messages before blocking."""
        from bot.memory.relationship_scorer import RelationshipLevel, RelationshipTier

        mock_level = RelationshipLevel(score=9.5, tier=RelationshipTier.INTIMATE, level=9)
        scheduler_with_deps._relationship_scorer.compute = AsyncMock(return_value=mock_level)

        user_id = 88
        for _ in range(4):
            assert await scheduler_with_deps._check_anti_spam(user_id) is True
            scheduler_with_deps._record_send(user_id)

        assert await scheduler_with_deps._check_anti_spam(user_id) is False


# ---------------------------------------------------------------------------
# Personalized prompt includes user facts
# ---------------------------------------------------------------------------


class TestPersonalizedPrompt:
    @pytest.mark.asyncio
    async def test_personalized_prompt_includes_user_facts(
        self, scheduler_with_deps, mock_langfuse
    ):
        """send_proactive_message appends profile interests/likes to prompt hint."""
        from bot.memory.models import UserProfile

        profile = UserProfile(user_id=55)
        profile.interests = ["программирование", "кино"]
        profile.likes = ["кофе"]

        scheduler_with_deps._profile_builder.get_profile = AsyncMock(return_value=profile)
        scheduler_with_deps._relationship_scorer.compute = AsyncMock(
            return_value=MagicMock(tier=MagicMock())
        )

        captured_messages: list[list[dict]] = []
        original_generate = scheduler_with_deps._llm.generate

        async def capturing_generate(messages):
            captured_messages.append(messages)
            return await original_generate(messages)

        scheduler_with_deps._llm.generate = capturing_generate

        with (
            patch(
                "bot.adapters.proactive_scheduler.get_system_prompt",
                return_value="system",
            ),
            patch(
                "bot.adapters.proactive_scheduler.get_langfuse_service",
                return_value=mock_langfuse,
            ),
            patch.object(scheduler_with_deps, "_is_quiet_hours", return_value=False),
            patch.object(scheduler_with_deps, "_check_anti_spam", new=AsyncMock(return_value=True)),
        ):
            await scheduler_with_deps.send_proactive_message(55, "утреннее приветствие")

        assert captured_messages, "LLM was not called"
        prompt_content = " ".join(
            m["content"] for m in captured_messages[0] if m.get("role") == "system"
        )
        assert "программирование" in prompt_content
        assert "кофе" in prompt_content

    @pytest.mark.asyncio
    async def test_prompt_unchanged_when_profile_empty(self, scheduler_with_deps, mock_langfuse):
        """When profile has no interests/likes, hint is not modified."""
        from bot.memory.models import UserProfile

        profile = UserProfile(user_id=66)
        # No interests or likes
        scheduler_with_deps._profile_builder.get_profile = AsyncMock(return_value=profile)

        captured_messages: list[list[dict]] = []
        original_generate = scheduler_with_deps._llm.generate

        async def capturing_generate(messages):
            captured_messages.append(messages)
            return await original_generate(messages)

        scheduler_with_deps._llm.generate = capturing_generate

        with (
            patch(
                "bot.adapters.proactive_scheduler.get_system_prompt",
                return_value="system",
            ),
            patch(
                "bot.adapters.proactive_scheduler.get_langfuse_service",
                return_value=mock_langfuse,
            ),
            patch.object(scheduler_with_deps, "_is_quiet_hours", return_value=False),
            patch.object(scheduler_with_deps, "_check_anti_spam", new=AsyncMock(return_value=True)),
        ):
            await scheduler_with_deps.send_proactive_message(66, "вечерний вопрос")

        prompt_content = " ".join(
            m["content"] for m in captured_messages[0] if m.get("role") == "system"
        )
        assert "Вспомни" not in prompt_content


# ---------------------------------------------------------------------------
# Milestone messages
# ---------------------------------------------------------------------------


class TestMilestoneMessage:
    @pytest.mark.asyncio
    async def test_milestone_message_sent_on_tier_up(
        self, scheduler_with_deps, mock_delivery, mock_langfuse
    ):
        """send_milestone_message calls LLM and delivers message."""
        from bot.memory.relationship_scorer import RelationshipTier

        with (
            patch(
                "bot.adapters.proactive_scheduler.get_system_prompt",
                return_value="system",
            ),
            patch(
                "bot.adapters.proactive_scheduler.get_langfuse_service",
                return_value=mock_langfuse,
            ),
            patch.object(scheduler_with_deps, "_is_quiet_hours", return_value=False),
        ):
            await scheduler_with_deps.send_milestone_message(
                user_id=10,
                old_tier=RelationshipTier.ACQUAINTANCE,
                new_tier=RelationshipTier.FRIEND,
            )

        mock_delivery.send_text.assert_called_once_with(chat_id=10, text="Привет, как дела?")

    @pytest.mark.asyncio
    async def test_milestone_max_1_per_day(self, scheduler_with_deps, mock_delivery, mock_langfuse):
        """Second milestone message on same day is silently skipped."""
        from bot.memory.relationship_scorer import RelationshipTier

        with (
            patch(
                "bot.adapters.proactive_scheduler.get_system_prompt",
                return_value="system",
            ),
            patch(
                "bot.adapters.proactive_scheduler.get_langfuse_service",
                return_value=mock_langfuse,
            ),
            patch.object(scheduler_with_deps, "_is_quiet_hours", return_value=False),
        ):
            await scheduler_with_deps.send_milestone_message(
                user_id=20,
                old_tier=RelationshipTier.ACQUAINTANCE,
                new_tier=RelationshipTier.FRIEND,
            )
            # Second call same day
            await scheduler_with_deps.send_milestone_message(
                user_id=20,
                old_tier=RelationshipTier.FRIEND,
                new_tier=RelationshipTier.CLOSE_FRIEND,
            )

        # Only one delivery despite two tier-ups
        assert mock_delivery.send_text.call_count == 1

    @pytest.mark.asyncio
    async def test_milestone_skipped_quiet_hours(
        self, scheduler_with_deps, mock_delivery, mock_langfuse
    ):
        """Milestone message is skipped during quiet hours."""
        from bot.memory.relationship_scorer import RelationshipTier

        with (
            patch(
                "bot.adapters.proactive_scheduler.get_system_prompt",
                return_value="system",
            ),
            patch(
                "bot.adapters.proactive_scheduler.get_langfuse_service",
                return_value=mock_langfuse,
            ),
            patch.object(scheduler_with_deps, "_is_quiet_hours", return_value=True),
        ):
            await scheduler_with_deps.send_milestone_message(
                user_id=30,
                old_tier=RelationshipTier.ACQUAINTANCE,
                new_tier=RelationshipTier.FRIEND,
            )

        mock_delivery.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_milestone_skipped_when_no_llm(self, mock_delivery):
        """Milestone message is silently skipped when LLM is not configured."""
        from bot.memory.relationship_scorer import RelationshipTier

        with patch("bot.adapters.proactive_scheduler.AsyncIOScheduler"):
            from bot.adapters.proactive_scheduler import ProactiveScheduler

            scheduler = ProactiveScheduler(delivery=mock_delivery, llm=None)

        await scheduler.send_milestone_message(
            user_id=40,
            old_tier=RelationshipTier.ACQUAINTANCE,
            new_tier=RelationshipTier.FRIEND,
        )

        mock_delivery.send_text.assert_not_called()


# ---------------------------------------------------------------------------
# Constructor injection — no singleton calls
# ---------------------------------------------------------------------------


class TestConstructorInjection:
    def test_constructor_injection_replaces_singletons(self, mock_delivery, mock_llm):
        """ProactiveScheduler stores injected deps, not singletons."""
        with patch("bot.adapters.proactive_scheduler.AsyncIOScheduler"):
            from bot.adapters.proactive_scheduler import ProactiveScheduler

            mock_db = MagicMock()
            mock_em = MagicMock()
            mock_scorer = MagicMock()
            mock_pb = MagicMock()

            s = ProactiveScheduler(
                delivery=mock_delivery,
                llm=mock_llm,
                db_client=mock_db,
                episode_manager=mock_em,
                relationship_scorer=mock_scorer,
                profile_builder=mock_pb,
            )

        assert s._llm is mock_llm
        assert s._db_client is mock_db
        assert s._episode_manager is mock_em
        assert s._relationship_scorer is mock_scorer
        assert s._profile_builder is mock_pb

    def test_all_optional_deps_default_to_none(self, mock_delivery):
        """All optional constructor params default to None."""
        with patch("bot.adapters.proactive_scheduler.AsyncIOScheduler"):
            from bot.adapters.proactive_scheduler import ProactiveScheduler

            s = ProactiveScheduler(delivery=mock_delivery)

        assert s._llm is None
        assert s._db_client is None
        assert s._episode_manager is None
        assert s._relationship_scorer is None
        assert s._profile_builder is None
        assert s._character is None

    @pytest.mark.asyncio
    async def test_send_returns_false_when_no_llm(self, mock_delivery):
        """send_proactive_message returns False immediately when LLM is None."""
        with patch("bot.adapters.proactive_scheduler.AsyncIOScheduler"):
            from bot.adapters.proactive_scheduler import ProactiveScheduler

            s = ProactiveScheduler(delivery=mock_delivery, llm=None)

        with patch.object(s, "_is_quiet_hours", return_value=False):
            result = await s.send_proactive_message(123, "test")

        assert result is False
        mock_delivery.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_active_user_ids_returns_empty_when_no_db(self, scheduler_no_deps):
        """_get_active_user_ids returns [] when db_client is None."""
        result = await scheduler_no_deps._get_active_user_ids()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_last_message_time_returns_none_when_no_db(self, scheduler_no_deps):
        """_get_last_message_time returns None when db_client is None."""
        result = await scheduler_no_deps._get_last_message_time(user_id=1)
        assert result is None


# ---------------------------------------------------------------------------
# Idle check uses per-user threshold
# ---------------------------------------------------------------------------


class TestIdleCheckPerUserThreshold:
    @pytest.mark.asyncio
    async def test_idle_check_respects_acquaintance_threshold(self, scheduler_with_deps):
        """ACQUAINTANCE user idle 13h triggers message; idle 11h does not."""
        from bot.memory.relationship_scorer import RelationshipLevel, RelationshipTier

        mock_level = RelationshipLevel(score=1.0, tier=RelationshipTier.ACQUAINTANCE, level=1)
        scheduler_with_deps._relationship_scorer.compute = AsyncMock(return_value=mock_level)
        scheduler_with_deps._get_active_user_ids = AsyncMock(return_value=[111, 222])
        scheduler_with_deps.send_proactive_message = AsyncMock(return_value=True)

        now = datetime.now(tz=UTC)

        async def mock_last_msg(uid: int) -> datetime | None:
            if uid == 111:
                return now - timedelta(hours=13)  # > 12h → should trigger
            return now - timedelta(hours=11)  # < 12h → should NOT trigger

        scheduler_with_deps._get_last_message_time = AsyncMock(side_effect=mock_last_msg)

        await scheduler_with_deps._idle_check()

        # Only user 111 should receive a message
        scheduler_with_deps.send_proactive_message.assert_called_once_with(
            111, "давно не общались, скучаю"
        )


# ---------------------------------------------------------------------------
# Cross-memory references in FactExtractorService
# ---------------------------------------------------------------------------


class TestCrossMemoryReferences:
    @pytest.mark.asyncio
    async def test_cross_memory_search_populates_related(self):
        """High-similarity search results populate fact.related_memories."""
        from bot.memory.fact_extractor import RELATED_MEMORY_THRESHOLD, FactExtractorService
        from bot.memory.models import MemoryFact, UserProfile

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"facts": [{"content": "Любит Python", "category": "preference",'
                ' "memory_type": "like", "importance": 1.2, "emotional_valence": 0.3,'
                ' "tags": ["python"]}]}'
            )
        )

        # Memory with score above threshold
        high_score_mem = MemoryFact(content="Занимается программированием", user_id=1)
        high_score_mem.fact_id = "abc123"
        high_score_mem.metadata = {"score": RELATED_MEMORY_THRESHOLD + 0.1}

        mock_mem0 = MagicMock()
        mock_mem0.search = AsyncMock(return_value=[high_score_mem])

        service = FactExtractorService(llm=mock_llm, mem0_service=mock_mem0)
        facts = await service.extract(
            user_message="Я люблю Python",
            bot_response="Отлично!",
            existing_profile=UserProfile(user_id=1),
            user_id=1,
        )

        assert len(facts) == 1
        assert "abc123" in facts[0].related_memories

    @pytest.mark.asyncio
    async def test_cross_memory_below_threshold_ignored(self):
        """Search results with score at or below threshold are ignored."""
        from bot.memory.fact_extractor import RELATED_MEMORY_THRESHOLD, FactExtractorService
        from bot.memory.models import MemoryFact

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"facts": [{"content": "Любит Python", "category": "preference",'
                ' "memory_type": "like", "importance": 1.0, "emotional_valence": 0.0,'
                ' "tags": []}]}'
            )
        )

        # Memory with score exactly at threshold (not above)
        low_score_mem = MemoryFact(content="Что-то похожее", user_id=1)
        low_score_mem.fact_id = "xyz789"
        low_score_mem.metadata = {"score": RELATED_MEMORY_THRESHOLD}  # exactly at, not above

        mock_mem0 = MagicMock()
        mock_mem0.search = AsyncMock(return_value=[low_score_mem])

        service = FactExtractorService(llm=mock_llm, mem0_service=mock_mem0)
        facts = await service.extract(
            user_message="Я люблю Python",
            bot_response="Класс!",
            user_id=1,
        )

        assert len(facts) == 1
        assert facts[0].related_memories == []

    @pytest.mark.asyncio
    async def test_cross_memory_graceful_on_search_failure(self):
        """Search failure is swallowed — facts still returned without related_memories."""
        from bot.memory.fact_extractor import FactExtractorService

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"facts": [{"content": "Любит кофе", "category": "preference",'
                ' "memory_type": "like", "importance": 1.0, "emotional_valence": 0.1,'
                ' "tags": []}]}'
            )
        )

        mock_mem0 = MagicMock()
        mock_mem0.search = AsyncMock(side_effect=Exception("mem0 unavailable"))

        service = FactExtractorService(llm=mock_llm, mem0_service=mock_mem0)
        facts = await service.extract(
            user_message="Хочу кофе",
            bot_response="Понятно!",
            user_id=1,
        )

        # Facts still extracted despite mem0 failure
        assert len(facts) == 1
        assert facts[0].related_memories == []

    @pytest.mark.asyncio
    async def test_cross_memory_skipped_when_user_id_zero(self):
        """Cross-memory search is skipped when user_id is 0 (unknown)."""
        from bot.memory.fact_extractor import FactExtractorService

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"facts": [{"content": "Любит чай", "category": "preference",'
                ' "memory_type": "like", "importance": 1.0, "emotional_valence": 0.0,'
                ' "tags": []}]}'
            )
        )

        mock_mem0 = MagicMock()
        mock_mem0.search = AsyncMock()

        service = FactExtractorService(llm=mock_llm, mem0_service=mock_mem0)
        facts = await service.extract(
            user_message="Чай вкусный",
            bot_response="Да!",
            user_id=0,  # unknown user
        )

        # mem0.search should NOT be called for user_id=0
        mock_mem0.search.assert_not_called()
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_cross_memory_skipped_when_no_facts(self):
        """Cross-memory search is not called when extraction returns no facts."""
        from bot.memory.fact_extractor import FactExtractorService

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=MagicMock(content='{"facts": []}'))

        mock_mem0 = MagicMock()
        mock_mem0.search = AsyncMock()

        service = FactExtractorService(llm=mock_llm, mem0_service=mock_mem0)
        facts = await service.extract(
            user_message="Привет",
            bot_response="Привет!",
            user_id=1,
        )

        mock_mem0.search.assert_not_called()
        assert facts == []
