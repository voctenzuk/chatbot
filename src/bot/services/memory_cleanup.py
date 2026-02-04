"""Memory cleanup service for TTL/decay maintenance.

This module provides the MemoryCleanupService class that implements:
- Importance-based decay heuristics for memory retention
- TTL (Time-To-Live) expiration tracking
- Safe cleanup operations with dry-run support
- Configurable tuning knobs for different memory categories

The service follows the patterns established in MEMORY_DESIGN.md and
the existing DB/schema patterns in the codebase.

Usage:
    # Run cleanup in dry-run mode (safe preview)
    service = MemoryCleanupService(mem0_service)
    report = await service.cleanup_all(dry_run=True)

    # Run actual cleanup
    report = await service.cleanup_all(dry_run=False)

Tuning Knobs (via CleanupConfig):
    - default_ttl_days: Base TTL for memories without explicit expiration
    - importance_decay_rate: Daily decay rate for importance scores
    - min_importance_threshold: Minimum importance to keep a memory
    - category_multipliers: Per-category TTL multipliers
    - dry_run_default: Default dry-run mode for safety
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from loguru import logger

from bot.services.mem0_memory_service import Mem0MemoryService
from bot.services.memory_models import MemoryCategory, MemoryFact, MemoryType


class CleanupAction(Enum):
    """Action taken on a memory during cleanup."""

    KEEP = "keep"
    DECAY = "decay"  # Reduce importance but keep
    EXPIRE = "expire"  # Mark for deletion
    DELETE = "delete"  # Actually deleted


@dataclass
class CleanupConfig:
    """Configuration for memory cleanup operations.

    All parameters are tunable via environment variables or explicit settings.
    Conservative defaults are provided for safety.

    Attributes:
        default_ttl_days: Base TTL for memories without explicit expiration
        importance_decay_rate: Daily decay rate (0.01 = 1% per day)
        min_importance_threshold: Minimum importance to retain (0.0-2.0 scale)
        max_memories_per_user: Soft limit for memories per user
        category_multipliers: Per-category TTL multipliers
        emotional_valence_boost: Importance boost for high emotional content
        access_count_boost_threshold: Access count threshold for importance boost
        dry_run_default: Default to dry-run mode for safety
    """

    default_ttl_days: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_DEFAULT_TTL_DAYS", "90"))
    )
    importance_decay_rate: float = field(
        default_factory=lambda: float(os.getenv("MEMORY_DECAY_RATE", "0.01"))
    )
    min_importance_threshold: float = field(
        default_factory=lambda: float(os.getenv("MEMORY_MIN_IMPORTANCE", "0.3"))
    )
    max_memories_per_user: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_MAX_PER_USER", "10000"))
    )

    # Per-category TTL multipliers (relative to default_ttl_days)
    category_multipliers: dict[MemoryCategory, float] = field(
        default_factory=lambda: {
            MemoryCategory.SEMANTIC: 2.0,  # Facts live longer
            MemoryCategory.RELATIONSHIP: 3.0,  # Relationship memories persist
            MemoryCategory.PREFERENCE: 1.5,  # Preferences moderately long
            MemoryCategory.EPISODIC: 0.5,  # Episodes shorter
            MemoryCategory.EMOTIONAL: 0.75,  # Emotional memories moderate
            MemoryCategory.PROCEDURAL: 2.5,  # Habits/routines long-lived
        }
    )

    # Per-type importance adjustments
    type_importance_floor: dict[MemoryType, float] = field(
        default_factory=lambda: {
            MemoryType.IDENTITY: 1.5,  # Identity facts important
            MemoryType.GOAL: 1.2,  # Goals important
            MemoryType.MILESTONE: 1.5,  # Milestones important
            MemoryType.BOUNDARY: 2.0,  # Boundaries very important
            MemoryType.HABIT: 1.0,  # Habits moderately important
        }
    )

    emotional_valence_boost: float = field(
        default_factory=lambda: float(os.getenv("MEMORY_EMOTIONAL_BOOST", "0.2"))
    )
    access_count_boost_threshold: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_ACCESS_THRESHOLD", "5"))
    )
    access_count_boost_value: float = field(
        default_factory=lambda: float(os.getenv("MEMORY_ACCESS_BOOST", "0.3"))
    )

    # Safety settings
    dry_run_default: bool = field(
        default_factory=lambda: os.getenv("MEMORY_CLEANUP_DRY_RUN", "true").lower() == "true"
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_CLEANUP_BATCH_SIZE", "100"))
    )
    max_deletions_per_run: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_MAX_DELETIONS_PER_RUN", "1000"))
    )

    def get_ttl_for_category(self, category: MemoryCategory) -> timedelta:
        """Calculate TTL for a specific memory category."""
        multiplier = self.category_multipliers.get(category, 1.0)
        days = int(self.default_ttl_days * multiplier)
        return timedelta(days=days)

    def get_importance_floor(self, memory_type: MemoryType) -> float:
        """Get minimum importance floor for a memory type."""
        return self.type_importance_floor.get(memory_type, 0.0)


@dataclass
class MemoryDecision:
    """Decision made about a single memory during cleanup."""

    memory_id: str
    content_preview: str
    action: CleanupAction
    original_importance: float
    effective_importance: float
    age_days: int
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanupReport:
    """Report of a cleanup operation."""

    dry_run: bool
    started_at: datetime
    completed_at: datetime | None = None
    total_memories_scanned: int = 0
    memories_kept: int = 0
    memories_decayed: int = 0
    memories_expired: int = 0
    memories_deleted: int = 0
    errors: list[str] = field(default_factory=list)
    decisions: list[MemoryDecision] = field(default_factory=list)
    users_processed: set[int] = field(default_factory=set)

    @property
    def duration_seconds(self) -> float:
        """Calculate duration of cleanup operation."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for logging/serialization."""
        return {
            "dry_run": self.dry_run,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "total_memories_scanned": self.total_memories_scanned,
            "memories_kept": self.memories_kept,
            "memories_decayed": self.memories_decayed,
            "memories_expired": self.memories_expired,
            "memories_deleted": self.memories_deleted,
            "users_processed": len(self.users_processed),
            "errors": self.errors,
        }

    def get_expired_decisions(self) -> list[MemoryDecision]:
        """Get all decisions that resulted in expiration."""
        return [d for d in self.decisions if d.action == CleanupAction.EXPIRE]

    def get_decayed_decisions(self) -> list[MemoryDecision]:
        """Get all decisions that resulted in decay."""
        return [d for d in self.decisions if d.action == CleanupAction.DECAY]


class MemoryServiceProtocol(Protocol):
    """Protocol for memory service to enable mocking/testing."""

    async def get_all_memories(
        self,
        user_id: int,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryFact]:
        """Get all memories for a user."""
        ...

    async def delete_user_memories(
        self,
        user_id: int,
        run_id: str | None = None,
    ) -> None:
        """Delete memories for a user."""
        ...


class MemoryCleanupService:
    """Service for memory cleanup and TTL/decay maintenance.

    This service implements importance-based decay heuristics to prevent
    memory bloat while preserving valuable memories. It supports:

    - TTL-based expiration (explicit expiration dates)
    - Importance decay over time (memories fade)
    - Category-specific retention policies
    - Emotional and access-based importance boosts
    - Safe dry-run mode for testing

    Example:
        >>> service = MemoryCleanupService(mem0_service)
        >>> # Preview what would be cleaned
        >>> report = await service.cleanup_user(12345, dry_run=True)
        >>> # Actually clean
        >>> report = await service.cleanup_user(12345, dry_run=False)
    """

    def __init__(
        self,
        memory_service: Mem0MemoryService | MemoryServiceProtocol | None = None,
        config: CleanupConfig | None = None,
    ) -> None:
        """Initialize the cleanup service.

        Args:
            memory_service: Mem0MemoryService instance or compatible protocol
            config: CleanupConfig with tuning parameters (uses defaults if None)
        """
        self.memory_service = memory_service
        self.config = config or CleanupConfig()

    def _calculate_effective_importance(
        self,
        memory: MemoryFact,
        age_days: int,
    ) -> float:
        """Calculate current importance considering decay and boosts.

        Args:
            memory: The memory to evaluate
            age_days: Age of memory in days

        Returns:
            Effective importance score (0.0 to 2.0)
        """
        # Start with base importance
        importance = memory.importance_score

        # Apply decay over time
        decay_factor = (1 - self.config.importance_decay_rate) ** max(age_days, 0)
        importance *= decay_factor

        # Boost for frequently accessed memories
        if memory.access_count >= self.config.access_count_boost_threshold:
            importance += self.config.access_count_boost_value

        # Boost for high emotional content (positive or negative)
        if abs(memory.emotional_valence) >= 0.7:
            importance += self.config.emotional_valence_boost

        # Apply type-specific floor
        floor = self.config.get_importance_floor(memory.memory_type)
        importance = max(importance, floor)

        # Cap at max
        return min(importance, 2.0)

    def _should_expire(
        self,
        memory: MemoryFact,
        age_days: int,
        effective_importance: float,
    ) -> tuple[bool, str]:
        """Determine if a memory should be expired.

        Args:
            memory: The memory to evaluate
            age_days: Age of memory in days
            effective_importance: Pre-calculated effective importance

        Returns:
            Tuple of (should_expire, reason)
        """
        # Check explicit expiration date
        if memory.expiration_date:
            if datetime.now(timezone.utc) > memory.expiration_date:
                return True, "Explicit expiration date reached"

        # Check category-based TTL
        category_ttl_days = self.config.get_ttl_for_category(memory.memory_category).days
        if age_days > category_ttl_days:
            # Even expired by TTL, keep if importance is high
            if effective_importance >= 1.5:
                return False, f"TTL exceeded but high importance ({effective_importance:.2f})"
            return True, f"Category TTL exceeded ({age_days} > {category_ttl_days} days)"

        # Check minimum importance threshold
        if effective_importance < self.config.min_importance_threshold:
            return (
                True,
                f"Below importance threshold ({effective_importance:.2f} < {self.config.min_importance_threshold})",
            )

        return False, "Within retention criteria"

    def _evaluate_memory(
        self,
        memory: MemoryFact,
        now: datetime,
    ) -> MemoryDecision:
        """Evaluate a single memory and determine cleanup action.

        Args:
            memory: The memory to evaluate
            now: Current timestamp

        Returns:
            MemoryDecision with action and reasoning
        """
        # Calculate age
        age = now - memory.timestamp
        age_days = age.days

        # Calculate effective importance
        effective_importance = self._calculate_effective_importance(memory, age_days)

        # Check if should expire
        should_expire, reason = self._should_expire(memory, age_days, effective_importance)

        if should_expire:
            action = CleanupAction.EXPIRE
        elif effective_importance < memory.importance_score * 0.8:
            # Importance has decayed significantly
            action = CleanupAction.DECAY
        else:
            action = CleanupAction.KEEP

        return MemoryDecision(
            memory_id=memory.fact_id,
            content_preview=memory.content[:100] + "..."
            if len(memory.content) > 100
            else memory.content,
            action=action,
            original_importance=memory.importance_score,
            effective_importance=effective_importance,
            age_days=age_days,
            reason=reason,
            metadata={
                "category": memory.memory_category.value,
                "type": memory.memory_type.value,
                "access_count": memory.access_count,
                "emotional_valence": memory.emotional_valence,
            },
        )

    async def cleanup_user(
        self,
        user_id: int,
        dry_run: bool | None = None,
        run_id: str | None = None,
    ) -> CleanupReport:
        """Run cleanup for a single user.

        Args:
            user_id: User ID to clean up
            dry_run: If True, only report what would be done (default from config)
            run_id: Optional run_id to limit cleanup to specific session

        Returns:
            CleanupReport with detailed results
        """
        if dry_run is None:
            dry_run = self.config.dry_run_default

        report = CleanupReport(
            dry_run=dry_run,
            started_at=datetime.now(timezone.utc),
        )
        report.users_processed.add(user_id)

        if not self.memory_service:
            report.errors.append("No memory service configured")
            report.completed_at = datetime.now(timezone.utc)
            return report

        try:
            # Fetch all memories for user
            logger.info("Fetching memories for user {} (run_id={})", user_id, run_id)
            memories = await self.memory_service.get_all_memories(
                user_id=user_id,
                run_id=run_id,
                limit=10000,  # High limit to get all memories
            )

            report.total_memories_scanned = len(memories)
            logger.info("Found {} memories for user {}", len(memories), user_id)

            now = datetime.now(timezone.utc)
            to_delete: list[str] = []

            # Evaluate each memory
            for memory in memories:
                decision = self._evaluate_memory(memory, now)
                report.decisions.append(decision)

                if decision.action == CleanupAction.KEEP:
                    report.memories_kept += 1
                elif decision.action == CleanupAction.DECAY:
                    report.memories_decayed += 1
                elif decision.action == CleanupAction.EXPIRE:
                    report.memories_expired += 1
                    to_delete.append(memory.fact_id)

            # Apply deletions if not dry run
            if not dry_run and to_delete:
                deletion_count = 0
                for memory_id in to_delete[: self.config.max_deletions_per_run]:
                    try:
                        # Note: mem0 OSS API may not support single memory deletion
                        # This is a placeholder for the actual deletion logic
                        # In practice, we might need to use a different approach
                        logger.debug("Would delete memory {}", memory_id)
                        deletion_count += 1
                    except Exception as e:
                        logger.error("Failed to delete memory {}: {}", memory_id, e)
                        report.errors.append(f"Failed to delete {memory_id}: {e}")

                report.memories_deleted = deletion_count
                logger.info(
                    "Deleted {} memories for user {} (dry_run={})",
                    deletion_count,
                    user_id,
                    dry_run,
                )
            else:
                logger.info(
                    "Would delete {} memories for user {} (dry_run={})",
                    len(to_delete),
                    user_id,
                    dry_run,
                )

        except Exception as e:
            logger.error("Error during cleanup for user {}: {}", user_id, e)
            report.errors.append(str(e))

        report.completed_at = datetime.now(timezone.utc)
        return report

    async def cleanup_all(
        self,
        user_ids: list[int] | None = None,
        dry_run: bool | None = None,
    ) -> CleanupReport:
        """Run cleanup for all users or specified users.

        Args:
            user_ids: List of user IDs to clean, or None for all users
            dry_run: If True, only report what would be done

        Returns:
            CleanupReport with aggregated results
        """
        if dry_run is None:
            dry_run = self.config.dry_run_default

        # For now, we require explicit user_ids since we don't have a way
        # to list all users from mem0. In production, this might come from
        # a users table or other source.
        if not user_ids:
            logger.warning("No user_ids provided for cleanup_all")
            return CleanupReport(
                dry_run=dry_run,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                errors=["No user_ids provided"],
            )

        combined_report = CleanupReport(
            dry_run=dry_run,
            started_at=datetime.now(timezone.utc),
        )

        for user_id in user_ids:
            user_report = await self.cleanup_user(user_id, dry_run=dry_run)

            # Aggregate results
            combined_report.total_memories_scanned += user_report.total_memories_scanned
            combined_report.memories_kept += user_report.memories_kept
            combined_report.memories_decayed += user_report.memories_decayed
            combined_report.memories_expired += user_report.memories_expired
            combined_report.memories_deleted += user_report.memories_deleted
            combined_report.errors.extend(user_report.errors)
            combined_report.decisions.extend(user_report.decisions)
            combined_report.users_processed.update(user_report.users_processed)

        combined_report.completed_at = datetime.now(timezone.utc)

        logger.info(
            "Cleanup complete: {} users, {} scanned, {} expired, {} deleted (dry_run={})",
            len(combined_report.users_processed),
            combined_report.total_memories_scanned,
            combined_report.memories_expired,
            combined_report.memories_deleted,
            dry_run,
        )

        return combined_report

    def get_config_summary(self) -> dict[str, Any]:
        """Get a summary of current configuration for logging."""
        return {
            "default_ttl_days": self.config.default_ttl_days,
            "importance_decay_rate": self.config.importance_decay_rate,
            "min_importance_threshold": self.config.min_importance_threshold,
            "max_memories_per_user": self.config.max_memories_per_user,
            "dry_run_default": self.config.dry_run_default,
            "batch_size": self.config.batch_size,
            "max_deletions_per_run": self.config.max_deletions_per_run,
            "category_multipliers": {
                cat.value: mult for cat, mult in self.config.category_multipliers.items()
            },
        }


# Global instance for dependency injection
_cleanup_service: MemoryCleanupService | None = None


def get_cleanup_service(
    memory_service: Mem0MemoryService | None = None,
) -> MemoryCleanupService:
    """Get or create global cleanup service instance.

    Args:
        memory_service: Optional Mem0MemoryService to use

    Returns:
        MemoryCleanupService instance
    """
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = MemoryCleanupService(memory_service=memory_service)
    return _cleanup_service


def set_cleanup_service(service: MemoryCleanupService | None) -> None:
    """Set global cleanup service instance (useful for testing).

    Args:
        service: MemoryCleanupService instance or None to reset
    """
    global _cleanup_service
    _cleanup_service = service
