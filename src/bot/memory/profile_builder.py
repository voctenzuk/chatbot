"""User profile builder — aggregates extracted facts into a UserProfile."""

import asyncio
from typing import Any

from loguru import logger

from bot.memory.fact_extractor import ExtractedFact
from bot.memory.models import MemoryCategory, MemoryType, UserProfile


class UserProfileBuilder:
    """Builds and maintains UserProfile from extracted facts.

    Holds an in-memory per-user cache with per-user asyncio locks to
    prevent concurrent rebuilds for the same user.
    """

    def __init__(
        self,
        db_client: Any | None = None,
        relationship_scorer: Any | None = None,
    ) -> None:
        self._db_client = db_client
        self._relationship_scorer = relationship_scorer
        self._profiles: dict[int, UserProfile] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    def _get_lock(self, user_id: int) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    async def get_profile(self, user_id: int) -> UserProfile:
        """Return cached profile or rebuild from DB.

        Args:
            user_id: Telegram user ID.

        Returns:
            UserProfile for this user.
        """
        if user_id in self._profiles:
            return self._profiles[user_id]
        return await self.rebuild_from_db(user_id)

    async def rebuild_from_db(self, user_id: int) -> UserProfile:
        """Rebuild UserProfile from user_facts rows in DB.

        Args:
            user_id: Telegram user ID.

        Returns:
            Freshly built UserProfile (also updates cache).
        """
        profile = UserProfile(user_id=user_id)

        if self._db_client is None:
            self._profiles[user_id] = profile
            return profile

        try:
            rows = await self._db_client.get_user_facts(user_id)
            profile = _rows_to_profile(user_id, rows)
        except Exception as exc:
            logger.warning("Failed to rebuild profile from DB for user {}: {}", user_id, exc)

        self._profiles[user_id] = profile
        return profile

    async def update(self, user_id: int, facts: list[ExtractedFact]) -> UserProfile:
        """Merge new facts into in-memory profile cache and invalidate scorer cache.

        Args:
            user_id: Telegram user ID.
            facts: New extracted facts to merge.

        Returns:
            Updated UserProfile.
        """
        async with self._get_lock(user_id):
            profile = self._profiles.get(user_id) or UserProfile(user_id=user_id)
            profile = _merge_facts_into_profile(profile, facts)
            self._profiles[user_id] = profile

        if self._relationship_scorer is not None:
            try:
                self._relationship_scorer.invalidate(user_id)
            except Exception as exc:
                logger.warning("Failed to invalidate relationship scorer cache: {}", exc)

        return profile


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _rows_to_profile(user_id: int, rows: list[dict[str, Any]]) -> UserProfile:
    """Build a UserProfile from raw user_facts DB rows."""
    profile = UserProfile(user_id=user_id)
    for row in rows:
        _apply_row_to_profile(profile, row)
    return profile


def _apply_row_to_profile(profile: UserProfile, row: dict[str, Any]) -> None:
    """Mutate profile in-place from a single DB row."""
    content: str = row.get("content", "")
    category_str: str = row.get("category", "")
    memory_type_str: str = row.get("memory_type", "")

    try:
        category = MemoryCategory(category_str)
    except ValueError:
        category = MemoryCategory.SEMANTIC

    try:
        memory_type = MemoryType(memory_type_str)
    except ValueError:
        memory_type = MemoryType.FACT

    _apply_to_profile(profile, content, category, memory_type)


def _merge_facts_into_profile(profile: UserProfile, facts: list[ExtractedFact]) -> UserProfile:
    """Merge facts into profile in-place. Returns the same profile."""
    for fact in facts:
        _apply_to_profile(profile, fact.content, fact.category, fact.memory_type)
    return profile


def _apply_to_profile(
    profile: UserProfile,
    content: str,
    category: MemoryCategory,
    memory_type: MemoryType,
) -> None:
    """Apply a single fact to profile fields based on category and memory_type."""
    if memory_type == MemoryType.LIKE:
        if content not in profile.likes:
            profile.likes.append(content)
    elif memory_type == MemoryType.DISLIKE:
        if content not in profile.dislikes:
            profile.dislikes.append(content)
    elif memory_type == MemoryType.TOPIC_INTEREST:
        if content not in profile.interests:
            profile.interests.append(content)
    elif memory_type == MemoryType.COMMUNICATION_STYLE:
        profile.communication_style = content
    elif memory_type == MemoryType.INSIDE_JOKE:
        if content not in profile.shared_jokes:
            profile.shared_jokes.append(content)
    elif memory_type == MemoryType.IDENTITY and category == MemoryCategory.SEMANTIC:
        _apply_identity(profile, content)
    elif (
        memory_type in (MemoryType.HABIT, MemoryType.ROUTINE)
        and content not in profile.common_topics
    ):
        profile.common_topics.append(content)


def _apply_identity(profile: UserProfile, content: str) -> None:
    """Heuristically apply identity content to profile fields."""
    lower = content.lower()
    if any(kw in lower for kw in ("имя", "зовут", "меня зовут")):
        # Don't overwrite if already set
        if not profile.name:
            # Try to extract just the name value
            for prefix in ("имя пользователя —", "имя —", "зовут", "меня зовут"):
                if prefix in lower:
                    name_part = content[lower.index(prefix) + len(prefix) :].strip(" —:-")
                    if name_part:
                        profile.name = name_part.split()[0].capitalize()
                        return
            return  # could not reliably extract name from content
    elif any(kw in lower for kw in ("работает", "профессия", "программист", "занимается")):
        if not profile.occupation:
            profile.occupation = content
    elif (
        any(kw in lower for kw in ("живёт", "живет", "город", "из ", "в москве", "в питере"))
        and not profile.location
    ):
        profile.location = content
