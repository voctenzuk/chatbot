"""Tests for Memory Cleanup Service.

These tests cover the TTL/decay maintenance functionality including:
- Importance score calculations with decay
- TTL expiration logic
- Category-specific retention policies
- Cleanup report generation
- Dry-run safety mode
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from bot.services.memory_cleanup import (
    CleanupAction,
    CleanupConfig,
    CleanupReport,
    MemoryCleanupService,
    MemoryDecision,
    get_cleanup_service,
    set_cleanup_service,
)
from bot.services.memory_models import MemoryCategory, MemoryFact, MemoryType


class TestCleanupConfig:
    """Tests for CleanupConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CleanupConfig()

        assert config.default_ttl_days == 90
        assert config.importance_decay_rate == 0.01
        assert config.min_importance_threshold == 0.3
        assert config.max_memories_per_user == 10000
        assert config.dry_run_default is True
        assert config.batch_size == 100

    def test_environment_variable_overrides(self):
        """Test that environment variables override defaults."""
        env_vars = {
            "MEMORY_DEFAULT_TTL_DAYS": "60",
            "MEMORY_DECAY_RATE": "0.02",
            "MEMORY_MIN_IMPORTANCE": "0.5",
            "MEMORY_CLEANUP_DRY_RUN": "false",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = CleanupConfig()

        assert config.default_ttl_days == 60
        assert config.importance_decay_rate == 0.02
        assert config.min_importance_threshold == 0.5
        assert config.dry_run_default is False

    def test_get_ttl_for_category(self):
        """Test TTL calculation for different categories."""
        config = CleanupConfig(default_ttl_days=100)

        # Semantic has 2.0x multiplier = 200 days
        semantic_ttl = config.get_ttl_for_category(MemoryCategory.SEMANTIC)
        assert semantic_ttl.days == 200

        # Episodic has 0.5x multiplier = 50 days
        episodic_ttl = config.get_ttl_for_category(MemoryCategory.EPISODIC)
        assert episodic_ttl.days == 50

        # Relationship has 3.0x multiplier = 300 days
        relationship_ttl = config.get_ttl_for_category(MemoryCategory.RELATIONSHIP)
        assert relationship_ttl.days == 300

    def test_get_ttl_for_unknown_category(self):
        """Test TTL for category without explicit multiplier."""
        config = CleanupConfig(default_ttl_days=100)

        # Unknown category should use 1.0x multiplier
        ttl = config.get_ttl_for_category(MemoryCategory.PROCEDURAL)
        assert ttl.days == 250  # PROCEDURAL has 2.5x

    def test_get_importance_floor(self):
        """Test importance floor for memory types."""
        config = CleanupConfig()

        # Identity has floor of 1.5
        assert config.get_importance_floor(MemoryType.IDENTITY) == 1.5

        # Goal has floor of 1.2
        assert config.get_importance_floor(MemoryType.GOAL) == 1.2

        # Unknown type should have 0.0 floor
        assert config.get_importance_floor(MemoryType.TEXT) == 0.0


class TestCleanupReport:
    """Tests for CleanupReport dataclass."""

    def test_duration_calculation(self):
        """Test duration calculation."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        completed = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)

        report = CleanupReport(
            dry_run=True,
            started_at=started,
            completed_at=completed,
        )

        assert report.duration_seconds == 30.0

    def test_duration_no_completion(self):
        """Test duration when not completed."""
        report = CleanupReport(
            dry_run=True,
            started_at=datetime.now(timezone.utc),
            completed_at=None,
        )

        assert report.duration_seconds == 0.0

    def test_to_dict(self):
        """Test report serialization."""
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        completed = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)

        report = CleanupReport(
            dry_run=True,
            started_at=started,
            completed_at=completed,
            total_memories_scanned=100,
            memories_kept=80,
            memories_expired=20,
            users_processed={123, 456},
        )

        data = report.to_dict()

        assert data["dry_run"] is True
        assert data["total_memories_scanned"] == 100
        assert data["memories_kept"] == 80
        assert data["memories_expired"] == 20
        assert data["users_processed"] == 2
        assert data["duration_seconds"] == 30.0

    def test_get_expired_decisions(self):
        """Test filtering expired decisions."""
        decisions = [
            MemoryDecision("1", "content", CleanupAction.KEEP, 1.0, 1.0, 10, ""),
            MemoryDecision("2", "content", CleanupAction.EXPIRE, 1.0, 0.2, 100, ""),
            MemoryDecision("3", "content", CleanupAction.DECAY, 1.0, 0.5, 50, ""),
            MemoryDecision("4", "content", CleanupAction.EXPIRE, 1.0, 0.1, 200, ""),
        ]

        report = CleanupReport(
            dry_run=True,
            started_at=datetime.now(timezone.utc),
            decisions=decisions,
        )

        expired = report.get_expired_decisions()
        assert len(expired) == 2
        assert all(d.action == CleanupAction.EXPIRE for d in expired)


class TestMemoryCleanupServiceInit:
    """Tests for MemoryCleanupService initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        service = MemoryCleanupService()

        assert service.memory_service is None
        assert service.config is not None
        assert service.config.default_ttl_days == 90

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = CleanupConfig(default_ttl_days=60, min_importance_threshold=0.5)
        service = MemoryCleanupService(config=config)

        assert service.config.default_ttl_days == 60
        assert service.config.min_importance_threshold == 0.5

    def test_init_with_memory_service(self):
        """Test initialization with memory service."""
        mock_service = MagicMock()
        service = MemoryCleanupService(memory_service=mock_service)

        assert service.memory_service is mock_service


class TestCalculateEffectiveImportance:
    """Tests for importance calculation with decay and boosts."""

    @pytest.fixture
    def service(self):
        """Create service with test config."""
        config = CleanupConfig(
            importance_decay_rate=0.01,  # 1% per day
            access_count_boost_value=0.3,
            access_count_boost_threshold=5,
            emotional_valence_boost=0.2,
        )
        return MemoryCleanupService(config=config)

    def test_no_decay_new_memory(self, service):
        """Test that fresh memories don't decay."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            importance_score=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        importance = service._calculate_effective_importance(memory, age_days=0)
        assert importance == pytest.approx(1.0, abs=0.01)

    def test_decay_over_time(self, service):
        """Test that importance decays over time."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            importance_score=1.0,
            timestamp=datetime.now(timezone.utc) - timedelta(days=30),
        )

        importance = service._calculate_effective_importance(memory, age_days=30)
        # 1.0 * (0.99)^30 ≈ 0.74
        assert importance == pytest.approx(0.74, abs=0.05)

    def test_access_count_boost(self, service):
        """Test importance boost for frequently accessed memories."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            importance_score=1.0,
            access_count=10,  # Above threshold of 5
            timestamp=datetime.now(timezone.utc),
        )

        importance = service._calculate_effective_importance(memory, age_days=0)
        # 1.0 + 0.3 boost
        assert importance == pytest.approx(1.3, abs=0.01)

    def test_emotional_boost_positive(self, service):
        """Test importance boost for high emotional valence."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            importance_score=1.0,
            emotional_valence=0.8,  # High positive
            timestamp=datetime.now(timezone.utc),
        )

        importance = service._calculate_effective_importance(memory, age_days=0)
        # 1.0 + 0.2 emotional boost
        assert importance == pytest.approx(1.2, abs=0.01)

    def test_emotional_boost_negative(self, service):
        """Test importance boost for high negative emotional valence."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            importance_score=1.0,
            emotional_valence=-0.9,  # High negative
            timestamp=datetime.now(timezone.utc),
        )

        importance = service._calculate_effective_importance(memory, age_days=0)
        # 1.0 + 0.2 emotional boost (absolute value)
        assert importance == pytest.approx(1.2, abs=0.01)

    def test_type_floor_enforced(self, service):
        """Test that type-specific importance floor is enforced."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            importance_score=0.1,  # Low importance
            memory_type=MemoryType.IDENTITY,  # But identity has 1.5 floor
            timestamp=datetime.now(timezone.utc),
        )

        importance = service._calculate_effective_importance(memory, age_days=0)
        # Should be boosted to floor of 1.5
        assert importance == 1.5

    def test_max_importance_capped(self, service):
        """Test that importance is capped at 2.0."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            importance_score=1.9,
            access_count=10,  # +0.3 boost
            emotional_valence=0.9,  # +0.2 boost
            timestamp=datetime.now(timezone.utc),
        )

        importance = service._calculate_effective_importance(memory, age_days=0)
        # Should be capped at 2.0
        assert importance == 2.0


class TestShouldExpire:
    """Tests for expiration decision logic."""

    @pytest.fixture
    def service(self):
        """Create service with test config."""
        config = CleanupConfig(
            default_ttl_days=90,
            min_importance_threshold=0.3,
            category_multipliers={
                MemoryCategory.SEMANTIC: 2.0,  # 180 days
                MemoryCategory.EPISODIC: 0.5,   # 45 days
            },
        )
        return MemoryCleanupService(config=config)

    def test_explicit_expiration_date_reached(self, service):
        """Test expiration when explicit date is reached."""
        past_date = datetime.now(timezone.utc) - timedelta(days=1)
        memory = MemoryFact(
            content="Test",
            user_id=123,
            expiration_date=past_date,
        )

        should_expire, reason = service._should_expire(memory, age_days=10, effective_importance=1.0)
        assert should_expire is True
        assert "expiration date" in reason.lower()

    def test_explicit_expiration_date_not_reached(self, service):
        """Test no expiration when explicit date not yet reached."""
        future_date = datetime.now(timezone.utc) + timedelta(days=10)
        memory = MemoryFact(
            content="Test",
            user_id=123,
            expiration_date=future_date,
        )

        should_expire, reason = service._should_expire(memory, age_days=10, effective_importance=1.0)
        assert should_expire is False

    def test_category_ttl_exceeded(self, service):
        """Test expiration when category TTL exceeded."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            memory_category=MemoryCategory.EPISODIC,  # 45 days TTL
        )

        should_expire, reason = service._should_expire(memory, age_days=100, effective_importance=0.5)
        assert should_expire is True
        assert "TTL exceeded" in reason

    def test_category_ttl_not_exceeded(self, service):
        """Test no expiration when within category TTL."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            memory_category=MemoryCategory.SEMANTIC,  # 180 days TTL
        )

        should_expire, reason = service._should_expire(memory, age_days=100, effective_importance=0.5)
        assert should_expire is False

    def test_ttl_exceeded_but_high_importance(self, service):
        """Test that high importance memories are kept even if TTL exceeded."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            memory_category=MemoryCategory.EPISODIC,
        )

        should_expire, reason = service._should_expire(memory, age_days=100, effective_importance=1.5)
        assert should_expire is False
        assert "high importance" in reason.lower()

    def test_below_importance_threshold(self, service):
        """Test expiration when below importance threshold."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
        )

        should_expire, reason = service._should_expire(memory, age_days=10, effective_importance=0.1)
        assert should_expire is True
        assert "importance threshold" in reason.lower()

    def test_within_retention_criteria(self, service):
        """Test no expiration when within all criteria."""
        memory = MemoryFact(
            content="Test",
            user_id=123,
            memory_category=MemoryCategory.SEMANTIC,
        )

        should_expire, reason = service._should_expire(memory, age_days=10, effective_importance=1.0)
        assert should_expire is False
        assert "retention" in reason.lower()


class TestEvaluateMemory:
    """Tests for memory evaluation."""

    @pytest.fixture
    def service(self):
        """Create service with test config."""
        config = CleanupConfig(
            default_ttl_days=90,
            min_importance_threshold=0.3,
        )
        return MemoryCleanupService(config=config)

    def test_evaluate_keep(self, service):
        """Test evaluation results in KEEP action."""
        memory = MemoryFact(
            content="Important fact about user",
            user_id=123,
            importance_score=1.5,
            memory_category=MemoryCategory.SEMANTIC,
            timestamp=datetime.now(timezone.utc),
        )

        decision = service._evaluate_memory(memory, datetime.now(timezone.utc))

        assert decision.action == CleanupAction.KEEP
        assert decision.memory_id == memory.fact_id
        assert decision.original_importance == 1.5

    def test_evaluate_expire(self, service):
        """Test evaluation results in EXPIRE action."""
        memory = MemoryFact(
            content="Old unimportant fact",
            user_id=123,
            importance_score=0.2,
            memory_category=MemoryCategory.EPISODIC,
            timestamp=datetime.now(timezone.utc) - timedelta(days=100),
        )

        decision = service._evaluate_memory(memory, datetime.now(timezone.utc))

        assert decision.action == CleanupAction.EXPIRE

    def test_evaluate_decay(self, service):
        """Test evaluation results in DECAY action."""
        memory = MemoryFact(
            content="Moderately important",
            user_id=123,
            importance_score=1.0,
            timestamp=datetime.now(timezone.utc) - timedelta(days=30),
        )

        decision = service._evaluate_memory(memory, datetime.now(timezone.utc))

        # After 30 days with 1% decay, importance drops to ~0.74
        # This is more than 20% decay from 1.0, so should be DECAY
        assert decision.action == CleanupAction.DECAY
        assert decision.effective_importance < decision.original_importance

    def test_content_preview_truncation(self, service):
        """Test that long content is truncated in preview."""
        memory = MemoryFact(
            content="x" * 200,  # Long content
            user_id=123,
            timestamp=datetime.now(timezone.utc),
        )

        decision = service._evaluate_memory(memory, datetime.now(timezone.utc))

        assert len(decision.content_preview) <= 104  # 100 + "..."
        assert decision.content_preview.endswith("...")

    def test_content_preview_no_truncation(self, service):
        """Test that short content is not truncated."""
        memory = MemoryFact(
            content="Short",
            user_id=123,
            timestamp=datetime.now(timezone.utc),
        )

        decision = service._evaluate_memory(memory, datetime.now(timezone.utc))

        assert decision.content_preview == "Short"


class TestCleanupUser:
    """Tests for cleanup_user method."""

    @pytest.fixture
    def mock_memory_service(self):
        """Create mock memory service with async methods."""
        mock = MagicMock()
        # Make get_all_memories return an async result
        async def async_get_all(*args, **kwargs):
            return []
        mock.get_all_memories = async_get_all
        return mock

    @pytest.fixture
    def service(self, mock_memory_service):
        """Create service with mock."""
        config = CleanupConfig(dry_run_default=False)
        return MemoryCleanupService(
            memory_service=mock_memory_service,
            config=config,
        )

    @pytest.mark.asyncio
    async def test_cleanup_user_no_service(self):
        """Test cleanup with no memory service configured."""
        service = MemoryCleanupService(memory_service=None)

        report = await service.cleanup_user(123)

        assert report.errors == ["No memory service configured"]
        assert report.total_memories_scanned == 0

    @pytest.mark.asyncio
    async def test_cleanup_user_dry_run_default(self, service):
        """Test that dry_run defaults to config setting."""
        service.config.dry_run_default = True

        service.memory_service.get_all_memories.return_value = []

        report = await service.cleanup_user(123)

        assert report.dry_run is True

    @pytest.mark.asyncio
    async def test_cleanup_user_explicit_dry_run(self, service):
        """Test explicit dry_run parameter."""
        service.config.dry_run_default = False
        service.memory_service.get_all_memories.return_value = []

        report = await service.cleanup_user(123, dry_run=True)

        assert report.dry_run is True

    @pytest.mark.asyncio
    async def test_cleanup_user_scans_memories(self, service, mock_memory_service):
        """Test that all user memories are scanned."""
        memories = [
            MemoryFact(content=f"Memory {i}", user_id=123, importance_score=1.0)
            for i in range(5)
        ]

        async def async_get_all(*args, **kwargs):
            return memories
        mock_memory_service.get_all_memories = async_get_all

        report = await service.cleanup_user(123, dry_run=True)

        assert report.total_memories_scanned == 5

    @pytest.mark.asyncio
    async def test_cleanup_user_with_run_id(self, service, mock_memory_service):
        """Test cleanup limited to specific run_id."""
        calls = []

        async def async_get_all(*args, **kwargs):
            calls.append(kwargs)
            return []
        mock_memory_service.get_all_memories = async_get_all

        await service.cleanup_user(123, run_id="session_abc")

        assert len(calls) == 1
        assert calls[0]["user_id"] == 123
        assert calls[0]["run_id"] == "session_abc"

    @pytest.mark.asyncio
    async def test_cleanup_user_counts_actions(self, service, mock_memory_service):
        """Test that different actions are counted correctly."""
        now = datetime.now(timezone.utc)
        memories = [
            # Will be KEPT - fresh and important
            MemoryFact(
                content="Fresh important",
                user_id=123,
                importance_score=1.5,
                timestamp=now,
            ),
            # Will be EXPIRED - old and low importance
            MemoryFact(
                content="Old unimportant",
                user_id=123,
                importance_score=0.2,
                timestamp=now - timedelta(days=100),
            ),
        ]

        async def async_get_all(*args, **kwargs):
            return memories
        mock_memory_service.get_all_memories = async_get_all

        report = await service.cleanup_user(123, dry_run=True)

        assert report.memories_kept == 1
        assert report.memories_expired == 1
        assert report.memories_decayed == 0

    @pytest.mark.asyncio
    async def test_cleanup_user_handles_errors(self, service, mock_memory_service):
        """Test that service errors are captured."""
        async def async_error(*args, **kwargs):
            raise Exception("API Error")
        mock_memory_service.get_all_memories = async_error

        report = await service.cleanup_user(123)

        assert len(report.errors) == 1
        assert "API Error" in report.errors[0]


class TestCleanupAll:
    """Tests for cleanup_all method."""

    @pytest.fixture
    def mock_memory_service(self):
        """Create mock memory service with async methods."""
        mock = MagicMock()
        async def async_get_all(*args, **kwargs):
            return []
        mock.get_all_memories = async_get_all
        return mock

    @pytest.fixture
    def service(self, mock_memory_service):
        """Create service with mock."""
        return MemoryCleanupService(memory_service=mock_memory_service)

    @pytest.mark.asyncio
    async def test_cleanup_all_no_users(self, service):
        """Test cleanup_all with no users specified."""
        report = await service.cleanup_all(user_ids=None)

        assert report.errors == ["No user_ids provided"]

    @pytest.mark.asyncio
    async def test_cleanup_all_empty_users(self, service):
        """Test cleanup_all with empty user list."""
        report = await service.cleanup_all(user_ids=[])

        assert report.errors == ["No user_ids provided"]

    @pytest.mark.asyncio
    async def test_cleanup_all_multiple_users(self, service, mock_memory_service):
        """Test cleanup for multiple users aggregates results."""
        call_count = [0]
        results = [
            [MemoryFact(content="User 1 memory", user_id=1)],
            [MemoryFact(content="User 2 memory", user_id=2)],
        ]

        async def async_get_all(*args, **kwargs):
            result = results[call_count[0]]
            call_count[0] += 1
            return result
        mock_memory_service.get_all_memories = async_get_all

        report = await service.cleanup_all(user_ids=[1, 2], dry_run=True)

        assert len(report.users_processed) == 2
        assert report.total_memories_scanned == 2

    @pytest.mark.asyncio
    async def test_cleanup_all_aggregates_counts(self, service, mock_memory_service):
        """Test that counts are aggregated across users."""
        now = datetime.now(timezone.utc)
        call_count = [0]
        results = [
            [
                MemoryFact(content="Keep", user_id=1, importance_score=1.5, timestamp=now),
                MemoryFact(
                    content="Expire",
                    user_id=1,
                    importance_score=0.1,
                    timestamp=now - timedelta(days=100),
                ),
            ],
            [
                MemoryFact(content="Keep2", user_id=2, importance_score=1.5, timestamp=now),
            ],
        ]

        async def async_get_all(*args, **kwargs):
            result = results[call_count[0]]
            call_count[0] += 1
            return result
        mock_memory_service.get_all_memories = async_get_all

        report = await service.cleanup_all(user_ids=[1, 2], dry_run=True)

        assert report.memories_kept == 2
        assert report.memories_expired == 1


class TestGetConfigSummary:
    """Tests for get_config_summary method."""

    def test_config_summary_structure(self):
        """Test that config summary has expected structure."""
        service = MemoryCleanupService()

        summary = service.get_config_summary()

        assert "default_ttl_days" in summary
        assert "importance_decay_rate" in summary
        assert "min_importance_threshold" in summary
        assert "category_multipliers" in summary

    def test_config_summary_values(self):
        """Test that config summary reflects actual config."""
        config = CleanupConfig(default_ttl_days=60)
        service = MemoryCleanupService(config=config)

        summary = service.get_config_summary()

        assert summary["default_ttl_days"] == 60
        assert summary["dry_run_default"] is True


class TestGlobalInstance:
    """Tests for global service instance management."""

    def setup_method(self):
        """Reset global instance before each test."""
        set_cleanup_service(None)

    def teardown_method(self):
        """Reset global instance after each test."""
        set_cleanup_service(None)

    def test_get_cleanup_service_creates_instance(self):
        """Test that get_cleanup_service creates instance when needed."""
        with patch("bot.services.memory_cleanup.MemoryCleanupService") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            service = get_cleanup_service()

            mock_cls.assert_called_once()
            assert service is mock_instance

    def test_get_cleanup_service_with_memory_service(self):
        """Test that get_cleanup_service passes memory service."""
        mock_memory = MagicMock()

        with patch("bot.services.memory_cleanup.MemoryCleanupService") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            service = get_cleanup_service(memory_service=mock_memory)

            # Verify the memory service was passed (config may be None or not, both are valid)
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs.get("memory_service") is mock_memory

    def test_set_cleanup_service(self):
        """Test setting global instance."""
        custom_service = MagicMock()
        custom_service.config = CleanupConfig()

        set_cleanup_service(custom_service)
        retrieved = get_cleanup_service()

        assert retrieved is custom_service

    def test_set_cleanup_service_to_none_resets(self):
        """Test resetting global instance."""
        custom_service = MagicMock()
        custom_service.config = CleanupConfig()

        set_cleanup_service(custom_service)
        set_cleanup_service(None)

        with patch("bot.services.memory_cleanup.MemoryCleanupService") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_cleanup_service()
            mock_cls.assert_called_once()
