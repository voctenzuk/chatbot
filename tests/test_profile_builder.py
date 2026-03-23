"""Tests for UserProfileBuilder."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.memory.fact_extractor import ExtractedFact
from bot.memory.models import MemoryCategory, MemoryType, UserProfile
from bot.memory.profile_builder import (
    UserProfileBuilder,
    _apply_to_profile,
    _merge_facts_into_profile,
    _rows_to_profile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fact(
    content: str = "Тестовый факт",
    category: MemoryCategory = MemoryCategory.SEMANTIC,
    memory_type: MemoryType = MemoryType.FACT,
    importance: float = 1.0,
    emotional_valence: float = 0.0,
    tags: list[str] | None = None,
) -> ExtractedFact:
    return ExtractedFact(
        content=content,
        category=category,
        memory_type=memory_type,
        importance=importance,
        emotional_valence=emotional_valence,
        tags=tags or [],
    )


def _make_db_row(
    content: str = "Тестовый факт",
    category: str = "semantic",
    memory_type: str = "fact",
) -> dict:
    return {
        "id": "uuid-1",
        "user_id": 42,
        "content": content,
        "content_hash": "abc123",
        "category": category,
        "memory_type": memory_type,
        "importance": 1.0,
        "emotional_valence": 0.0,
        "tags": [],
    }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestUserProfileBuilderInit:
    def test_init_no_args(self) -> None:
        builder = UserProfileBuilder()
        assert builder._db_client is None
        assert builder._relationship_scorer is None
        assert builder._profiles == {}
        assert builder._locks == {}

    def test_init_with_db_client(self) -> None:
        db = MagicMock()
        builder = UserProfileBuilder(db_client=db)
        assert builder._db_client is db

    def test_init_with_scorer(self) -> None:
        scorer = MagicMock()
        builder = UserProfileBuilder(relationship_scorer=scorer)
        assert builder._relationship_scorer is scorer


# ---------------------------------------------------------------------------
# get_profile()
# ---------------------------------------------------------------------------


class TestGetProfile:
    @pytest.mark.asyncio
    async def test_returns_cached_profile(self) -> None:
        builder = UserProfileBuilder()
        cached = UserProfile(user_id=1, name="Кэш")
        builder._profiles[1] = cached

        profile = await builder.get_profile(1)
        assert profile is cached

    @pytest.mark.asyncio
    async def test_rebuilds_when_not_cached(self) -> None:
        db = MagicMock()
        db.get_user_facts = AsyncMock(return_value=[])
        builder = UserProfileBuilder(db_client=db)

        profile = await builder.get_profile(99)
        assert profile.user_id == 99
        db.get_user_facts.assert_called_once_with(99)

    @pytest.mark.asyncio
    async def test_no_db_returns_empty_profile(self) -> None:
        builder = UserProfileBuilder()
        profile = await builder.get_profile(5)
        assert profile.user_id == 5
        assert profile.name is None


# ---------------------------------------------------------------------------
# rebuild_from_db()
# ---------------------------------------------------------------------------


class TestRebuildFromDb:
    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_profile(self) -> None:
        db = MagicMock()
        db.get_user_facts = AsyncMock(return_value=[])
        builder = UserProfileBuilder(db_client=db)

        profile = await builder.rebuild_from_db(1)
        assert profile.user_id == 1
        assert profile.likes == []

    @pytest.mark.asyncio
    async def test_maps_like_row_to_likes(self) -> None:
        db = MagicMock()
        db.get_user_facts = AsyncMock(
            return_value=[_make_db_row("Любит кофе", category="preference", memory_type="like")]
        )
        builder = UserProfileBuilder(db_client=db)

        profile = await builder.rebuild_from_db(1)
        assert "Любит кофе" in profile.likes

    @pytest.mark.asyncio
    async def test_maps_dislike_row_to_dislikes(self) -> None:
        db = MagicMock()
        db.get_user_facts = AsyncMock(
            return_value=[
                _make_db_row("Не любит шум", category="preference", memory_type="dislike")
            ]
        )
        builder = UserProfileBuilder(db_client=db)

        profile = await builder.rebuild_from_db(1)
        assert "Не любит шум" in profile.dislikes

    @pytest.mark.asyncio
    async def test_db_error_returns_empty_profile(self) -> None:
        db = MagicMock()
        db.get_user_facts = AsyncMock(side_effect=RuntimeError("DB error"))
        builder = UserProfileBuilder(db_client=db)

        profile = await builder.rebuild_from_db(42)
        assert profile.user_id == 42
        assert profile.likes == []

    @pytest.mark.asyncio
    async def test_caches_rebuilt_profile(self) -> None:
        db = MagicMock()
        db.get_user_facts = AsyncMock(return_value=[])
        builder = UserProfileBuilder(db_client=db)

        profile = await builder.rebuild_from_db(7)
        assert builder._profiles[7] is profile


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


class TestUpdate:
    @pytest.mark.asyncio
    async def test_merges_like_fact(self) -> None:
        builder = UserProfileBuilder()
        builder._profiles[1] = UserProfile(user_id=1)

        facts = [_make_fact("Любит джаз", MemoryCategory.PREFERENCE, MemoryType.LIKE)]
        profile = await builder.update(1, facts)

        assert "Любит джаз" in profile.likes

    @pytest.mark.asyncio
    async def test_invalidates_scorer_cache(self) -> None:
        scorer = MagicMock()
        builder = UserProfileBuilder(relationship_scorer=scorer)
        builder._profiles[1] = UserProfile(user_id=1)

        await builder.update(1, [_make_fact()])

        scorer.invalidate.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_scorer_invalidation_failure_does_not_raise(self) -> None:
        scorer = MagicMock()
        scorer.invalidate.side_effect = RuntimeError("scorer error")
        builder = UserProfileBuilder(relationship_scorer=scorer)
        builder._profiles[1] = UserProfile(user_id=1)

        profile = await builder.update(1, [_make_fact()])
        assert profile is not None

    @pytest.mark.asyncio
    async def test_no_duplicate_likes(self) -> None:
        builder = UserProfileBuilder()
        builder._profiles[1] = UserProfile(user_id=1, likes=["Любит кофе"])

        facts = [_make_fact("Любит кофе", MemoryCategory.PREFERENCE, MemoryType.LIKE)]
        profile = await builder.update(1, facts)

        assert profile.likes.count("Любит кофе") == 1

    @pytest.mark.asyncio
    async def test_update_creates_profile_if_not_cached(self) -> None:
        builder = UserProfileBuilder()
        facts = [_make_fact("Любит рок", MemoryCategory.PREFERENCE, MemoryType.LIKE)]
        profile = await builder.update(99, facts)
        assert profile.user_id == 99
        assert "Любит рок" in profile.likes


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestRowsToProfile:
    def test_empty_rows(self) -> None:
        profile = _rows_to_profile(1, [])
        assert profile.user_id == 1
        assert profile.likes == []

    def test_multiple_like_rows(self) -> None:
        rows = [
            _make_db_row("Любит кофе", "preference", "like"),
            _make_db_row("Любит джаз", "preference", "like"),
        ]
        profile = _rows_to_profile(5, rows)
        assert "Любит кофе" in profile.likes
        assert "Любит джаз" in profile.likes

    def test_invalid_category_falls_back(self) -> None:
        row = _make_db_row(category="unknown_cat", memory_type="like")
        profile = _rows_to_profile(1, [row])
        assert isinstance(profile, UserProfile)

    def test_invalid_memory_type_falls_back(self) -> None:
        row = _make_db_row(category="preference", memory_type="unknown_type")
        profile = _rows_to_profile(1, [row])
        assert isinstance(profile, UserProfile)


class TestMergeFactsIntoProfile:
    def test_like_added(self) -> None:
        profile = UserProfile(user_id=1)
        facts = [_make_fact("Любит Python", MemoryCategory.PREFERENCE, MemoryType.LIKE)]
        result = _merge_facts_into_profile(profile, facts)
        assert "Любит Python" in result.likes

    def test_dislike_added(self) -> None:
        profile = UserProfile(user_id=1)
        facts = [_make_fact("Не любит спам", MemoryCategory.PREFERENCE, MemoryType.DISLIKE)]
        result = _merge_facts_into_profile(profile, facts)
        assert "Не любит спам" in result.dislikes

    def test_interest_added(self) -> None:
        profile = UserProfile(user_id=1)
        facts = [
            _make_fact("Интересуется ИИ", MemoryCategory.PREFERENCE, MemoryType.TOPIC_INTEREST)
        ]
        result = _merge_facts_into_profile(profile, facts)
        assert "Интересуется ИИ" in result.interests

    def test_inside_joke_added(self) -> None:
        profile = UserProfile(user_id=1)
        facts = [_make_fact("Про Барсика", MemoryCategory.RELATIONSHIP, MemoryType.INSIDE_JOKE)]
        result = _merge_facts_into_profile(profile, facts)
        assert "Про Барсика" in result.shared_jokes

    def test_communication_style_set(self) -> None:
        profile = UserProfile(user_id=1)
        facts = [
            _make_fact(
                "Неформальный стиль",
                MemoryCategory.PREFERENCE,
                MemoryType.COMMUNICATION_STYLE,
            )
        ]
        result = _merge_facts_into_profile(profile, facts)
        assert result.communication_style == "Неформальный стиль"

    def test_habit_added_to_common_topics(self) -> None:
        profile = UserProfile(user_id=1)
        facts = [
            _make_fact("Занимается спортом по утрам", MemoryCategory.PROCEDURAL, MemoryType.HABIT)
        ]
        result = _merge_facts_into_profile(profile, facts)
        assert "Занимается спортом по утрам" in result.common_topics


class TestApplyToProfile:
    def test_like(self) -> None:
        profile = UserProfile(user_id=1)
        _apply_to_profile(profile, "Кофе", MemoryCategory.PREFERENCE, MemoryType.LIKE)
        assert "Кофе" in profile.likes

    def test_dislike(self) -> None:
        profile = UserProfile(user_id=1)
        _apply_to_profile(profile, "Шум", MemoryCategory.PREFERENCE, MemoryType.DISLIKE)
        assert "Шум" in profile.dislikes

    def test_identity_with_name_keyword(self) -> None:
        profile = UserProfile(user_id=1)
        _apply_to_profile(
            profile,
            "Имя пользователя — Алексей",
            MemoryCategory.SEMANTIC,
            MemoryType.IDENTITY,
        )
        assert profile.name is not None

    def test_identity_does_not_overwrite_name(self) -> None:
        profile = UserProfile(user_id=1, name="Existing")
        _apply_to_profile(
            profile,
            "Имя пользователя — Новое",
            MemoryCategory.SEMANTIC,
            MemoryType.IDENTITY,
        )
        assert profile.name == "Existing"
