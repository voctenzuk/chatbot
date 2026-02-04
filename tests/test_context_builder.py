"""Tests for ContextBuilder service."""

import pytest
from datetime import datetime, timedelta

from bot.services.context_builder import (
    ContextBuilder,
    ContextAssemblyConfig,
    ContextPart,
    ConversationMessage,
    MessageRole,
    RunningSummary,
    get_context_builder,
    set_context_builder,
)
from bot.services.memory_models import (
    MemoryFact,
    MemoryCategory,
    MemoryType,
)


class TestConversationMessage:
    """Tests for ConversationMessage dataclass."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = ConversationMessage(
            role=MessageRole.USER,
            content="Hello!",
            message_id="msg_1",
        )

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert msg.message_id == "msg_1"
        assert isinstance(msg.timestamp, datetime)

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content="Hi there!",
        )

        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        msg = ConversationMessage(
            role=MessageRole.USER,
            content="Test message",
        )

        result = msg.to_dict()

        assert result == {"role": "user", "content": "Test message"}


class TestRunningSummary:
    """Tests for RunningSummary dataclass."""

    def test_create_summary(self):
        """Test creating a running summary."""
        summary = RunningSummary(
            content="User likes cats and lives in NY",
            message_count=50,
            version=2,
        )

        assert summary.content == "User likes cats and lives in NY"
        assert summary.message_count == 50
        assert summary.version == 2
        assert isinstance(summary.timestamp, datetime)

    def test_is_stale_true(self):
        """Test is_stale returns True for old summary."""
        old_time = datetime.now() - timedelta(hours=25)
        summary = RunningSummary(
            content="Old summary",
            timestamp=old_time,
        )

        assert summary.is_stale(max_age_hours=24) is True

    def test_is_stale_false(self):
        """Test is_stale returns False for recent summary."""
        recent_time = datetime.now() - timedelta(hours=1)
        summary = RunningSummary(
            content="Recent summary",
            timestamp=recent_time,
        )

        assert summary.is_stale(max_age_hours=24) is False


class TestContextAssemblyConfig:
    """Tests for ContextAssemblyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContextAssemblyConfig()

        assert config.max_total_tokens == 4000
        assert config.max_summary_tokens == 500
        assert config.max_recent_messages == 10
        assert config.max_semantic_memories == 5
        assert config.tokens_per_char == 0.25
        assert config.prune_by == "priority"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContextAssemblyConfig(
            max_total_tokens=2000,
            max_recent_messages=5,
            prune_by="recency",
            min_importance_score=0.5,
        )

        assert config.max_total_tokens == 2000
        assert config.max_recent_messages == 5
        assert config.prune_by == "recency"
        assert config.min_importance_score == 0.5

    def test_config_with_filters(self):
        """Test configuration with metadata filters."""
        config = ContextAssemblyConfig(
            include_categories=[MemoryCategory.SEMANTIC, MemoryCategory.PREFERENCE],
            exclude_types=[MemoryType.IMAGE],
        )

        assert config.include_categories == [MemoryCategory.SEMANTIC, MemoryCategory.PREFERENCE]
        assert config.exclude_types == [MemoryType.IMAGE]


class TestContextBuilderInit:
    """Tests for ContextBuilder initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        builder = ContextBuilder()

        assert builder.config is not None
        assert builder.config.max_total_tokens == 4000

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = ContextAssemblyConfig(max_total_tokens=1000)
        builder = ContextBuilder(config)

        assert builder.config.max_total_tokens == 1000


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens_basic(self):
        """Test basic token estimation."""
        builder = ContextBuilder()

        # 100 chars * 0.25 = 25 tokens
        tokens = builder._estimate_tokens("a" * 100)

        assert tokens == 25

    def test_estimate_tokens_custom_ratio(self):
        """Test token estimation with custom ratio."""
        config = ContextAssemblyConfig(tokens_per_char=0.5)
        builder = ContextBuilder(config)

        # 100 chars * 0.5 = 50 tokens
        tokens = builder._estimate_tokens("a" * 100)

        assert tokens == 50


class TestMemoryFiltering:
    """Tests for memory filtering."""

    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        return [
            MemoryFact(
                content="User likes cats",
                user_id=123,
                memory_category=MemoryCategory.PREFERENCE,
                memory_type=MemoryType.LIKE,
                importance_score=1.5,
            ),
            MemoryFact(
                content="User works at Google",
                user_id=123,
                memory_category=MemoryCategory.SEMANTIC,
                memory_type=MemoryType.FACT,
                importance_score=1.2,
            ),
            MemoryFact(
                content="User felt sad yesterday",
                user_id=123,
                memory_category=MemoryCategory.EMOTIONAL,
                memory_type=MemoryType.MOOD_STATE,
                importance_score=1.0,
            ),
        ]

    def test_filter_by_include_category(self, sample_memories):
        """Test filtering by included categories."""
        config = ContextAssemblyConfig(
            include_categories=[MemoryCategory.PREFERENCE, MemoryCategory.SEMANTIC]
        )
        builder = ContextBuilder(config)

        filtered = builder._filter_memories(sample_memories)

        assert len(filtered) == 2
        assert all(
            m.memory_category in [MemoryCategory.PREFERENCE, MemoryCategory.SEMANTIC]
            for m in filtered
        )

    def test_filter_by_exclude_category(self, sample_memories):
        """Test filtering by excluded categories."""
        config = ContextAssemblyConfig(exclude_categories=[MemoryCategory.EMOTIONAL])
        builder = ContextBuilder(config)

        filtered = builder._filter_memories(sample_memories)

        assert len(filtered) == 2
        assert all(m.memory_category != MemoryCategory.EMOTIONAL for m in filtered)

    def test_filter_by_importance_score(self, sample_memories):
        """Test filtering by minimum importance score."""
        config = ContextAssemblyConfig(min_importance_score=1.3)
        builder = ContextBuilder(config)

        filtered = builder._filter_memories(sample_memories)

        assert len(filtered) == 1
        assert filtered[0].content == "User likes cats"


class TestMemorySorting:
    """Tests for memory sorting strategies."""

    @pytest.fixture
    def unsorted_memories(self):
        """Create memories with varying importance for sorting tests."""
        base_time = datetime.now()
        return [
            MemoryFact(
                content="Low importance",
                user_id=123,
                importance_score=0.5,
                timestamp=base_time,
            ),
            MemoryFact(
                content="High importance",
                user_id=123,
                importance_score=2.0,
                timestamp=base_time - timedelta(hours=1),
            ),
            MemoryFact(
                content="Medium importance",
                user_id=123,
                importance_score=1.0,
                timestamp=base_time - timedelta(hours=2),
            ),
        ]

    def test_sort_by_priority(self, unsorted_memories):
        """Test sorting by priority (importance)."""
        config = ContextAssemblyConfig(prune_by="priority")
        builder = ContextBuilder(config)

        sorted_memories = builder._sort_memories(unsorted_memories)

        assert sorted_memories[0].content == "High importance"
        assert sorted_memories[1].content == "Medium importance"
        assert sorted_memories[2].content == "Low importance"

    def test_sort_by_recency(self, unsorted_memories):
        """Test sorting by recency."""
        config = ContextAssemblyConfig(prune_by="recency")
        builder = ContextBuilder(config)

        sorted_memories = builder._sort_memories(unsorted_memories)

        # Most recent first
        assert sorted_memories[0].content == "Low importance"
        assert sorted_memories[1].content == "High importance"
        assert sorted_memories[2].content == "Medium importance"

    def test_sort_by_relevance(self):
        """Test sorting by relevance (access count)."""
        memories = [
            MemoryFact(content="Never accessed", user_id=123, access_count=0),
            MemoryFact(content="Accessed 5 times", user_id=123, access_count=5),
            MemoryFact(content="Accessed 10 times", user_id=123, access_count=10),
        ]

        config = ContextAssemblyConfig(prune_by="relevance")
        builder = ContextBuilder(config)

        sorted_memories = builder._sort_memories(memories)

        assert sorted_memories[0].content == "Accessed 10 times"
        assert sorted_memories[1].content == "Accessed 5 times"
        assert sorted_memories[2].content == "Never accessed"


class TestPruneToTokenLimit:
    """Tests for token limit pruning."""

    def test_prune_within_limit(self):
        """Test no pruning when under limit."""
        builder = ContextBuilder()
        parts = [
            ContextPart(content="Short", source="test", priority=1, token_estimate=10),
            ContextPart(content="Also short", source="test", priority=1, token_estimate=15),
        ]

        pruned = builder._prune_to_token_limit(parts, 100)

        assert len(pruned) == 2

    def test_prune_exceeds_limit(self):
        """Test pruning when over limit."""
        builder = ContextBuilder()
        parts = [
            ContextPart(content="First", source="test", priority=1, token_estimate=30),
            ContextPart(content="Second", source="test", priority=1, token_estimate=30),
            ContextPart(content="Third", source="test", priority=1, token_estimate=30),
        ]

        pruned = builder._prune_to_token_limit(parts, 70)

        # Should include first two (60 tokens) but not third (would be 90)
        assert len(pruned) == 2

    def test_prune_empty_list(self):
        """Test pruning empty list."""
        builder = ContextBuilder()

        pruned = builder._prune_to_token_limit([], 100)

        assert pruned == []

    def test_prune_with_estimation(self):
        """Test pruning with automatic token estimation."""
        builder = ContextBuilder()
        # 40 chars * 0.25 = 10 tokens each
        parts = [
            ContextPart(content="a" * 40, source="test", priority=1),  # 10 tokens
            ContextPart(content="b" * 40, source="test", priority=1),  # 10 tokens
            ContextPart(content="c" * 40, source="test", priority=1),  # 10 tokens
        ]

        pruned = builder._prune_to_token_limit(parts, 25)

        # First two parts = 20 tokens, adding third would be 30
        assert len(pruned) == 2


class TestBuildSummaryPart:
    """Tests for building summary context part."""

    def test_build_with_summary(self):
        """Test building part with valid summary."""
        builder = ContextBuilder()
        summary = RunningSummary(
            content="User is a developer who likes Python",
            message_count=100,
            version=3,
        )

        part = builder.build_summary_part(summary)

        assert part is not None
        assert part.source == "summary"
        assert part.priority == 10
        assert "developer" in part.content
        assert part.metadata["message_count"] == 100

    def test_build_with_none_summary(self):
        """Test building part with None summary."""
        builder = ContextBuilder()

        part = builder.build_summary_part(None)

        assert part is None

    def test_build_truncates_long_summary(self):
        """Test that long summaries are truncated."""
        config = ContextAssemblyConfig(
            max_summary_tokens=10,
            tokens_per_char=1.0,  # 1 token per char for easy calculation
        )
        builder = ContextBuilder(config)
        summary = RunningSummary(content="a" * 100)

        part = builder.build_summary_part(summary)

        assert part is not None
        assert len(part.content) < 100
        assert part.content.endswith("...")


class TestBuildRecentMessagesPart:
    """Tests for building recent messages part."""

    def test_build_with_messages(self):
        """Test building part with messages."""
        builder = ContextBuilder()
        messages = [
            ConversationMessage(role=MessageRole.USER, content="Hello"),
            ConversationMessage(role=MessageRole.ASSISTANT, content="Hi!"),
        ]

        part = builder.build_recent_messages_part(messages)

        assert part is not None
        assert part.source == "recent_messages"
        assert "User: Hello" in part.content
        assert "Assistant: Hi!" in part.content

    def test_build_with_empty_messages(self):
        """Test building part with empty messages."""
        builder = ContextBuilder()

        part = builder.build_recent_messages_part([])

        assert part is None

    def test_build_respects_max_messages(self):
        """Test that max_recent_messages limit is respected."""
        config = ContextAssemblyConfig(max_recent_messages=2)
        builder = ContextBuilder(config)
        messages = [
            ConversationMessage(role=MessageRole.USER, content="First"),
            ConversationMessage(role=MessageRole.ASSISTANT, content="Second"),
            ConversationMessage(role=MessageRole.USER, content="Third"),
        ]

        part = builder.build_recent_messages_part(messages)

        assert part is not None
        assert "Second" in part.content
        assert "Third" in part.content
        assert "First" not in part.content  # Should be excluded (oldest)


class TestBuildSemanticMemoryPart:
    """Tests for building semantic memory part."""

    def test_build_with_memories(self):
        """Test building part with memories."""
        builder = ContextBuilder()
        memories = [
            MemoryFact(content="User likes Python", user_id=123),
            MemoryFact(content="User works remotely", user_id=123),
        ]

        part = builder.build_semantic_memory_part(memories, query="work")

        assert part is not None
        assert part.source == "semantic_memory"
        assert "User likes Python" in part.content
        assert "work" in part.content  # Query should be in header

    def test_build_with_empty_memories(self):
        """Test building part with empty memories."""
        builder = ContextBuilder()

        part = builder.build_semantic_memory_part([])

        assert part is None

    def test_build_with_filtered_memories(self):
        """Test that memories are filtered before building."""
        config = ContextAssemblyConfig(include_categories=[MemoryCategory.PREFERENCE])
        builder = ContextBuilder(config)
        memories = [
            MemoryFact(
                content="User likes cats",
                user_id=123,
                memory_category=MemoryCategory.PREFERENCE,
            ),
            MemoryFact(
                content="User works at Google",
                user_id=123,
                memory_category=MemoryCategory.SEMANTIC,
            ),
        ]

        part = builder.build_semantic_memory_part(memories)

        assert part is not None
        assert "cats" in part.content
        assert "Google" not in part.content  # Filtered out

    def test_build_shows_important_prefix(self):
        """Test that important memories get prefix."""
        builder = ContextBuilder()
        memories = [
            MemoryFact(
                content="Very important fact",
                user_id=123,
                importance_score=2.0,
            ),
        ]

        part = builder.build_semantic_memory_part(memories)

        assert part is not None
        assert "[Important]" in part.content

    def test_build_shows_emotional_prefix(self):
        """Test that emotional memories get prefix."""
        builder = ContextBuilder()
        memories = [
            MemoryFact(
                content="User felt happy",
                user_id=123,
                memory_category=MemoryCategory.EMOTIONAL,
            ),
        ]

        part = builder.build_semantic_memory_part(memories)

        assert part is not None
        assert "[Emotional]" in part.content


class TestAssemble:
    """Tests for the main assemble method."""

    def test_assemble_empty(self):
        """Test assembling with no inputs."""
        builder = ContextBuilder()

        result = builder.assemble()

        assert result == ""

    def test_assemble_with_all_parts(self):
        """Test assembling with all context sources."""
        builder = ContextBuilder()
        summary = RunningSummary(content="Summary here")
        messages = [ConversationMessage(role=MessageRole.USER, content="Hi")]
        memories = [MemoryFact(content="User fact", user_id=123)]

        result = builder.assemble(
            summary=summary,
            recent_messages=messages,
            semantic_memories=memories,
        )

        assert "Summary here" in result
        assert "User: Hi" in result
        assert "User fact" in result

    def test_assemble_deterministic_order(self):
        """Test that assemble produces deterministic ordering."""
        config = ContextAssemblyConfig(order=["summary", "semantic_memory", "recent_messages"])
        builder = ContextBuilder(config)
        summary = RunningSummary(content="SUMMARY")
        messages = [ConversationMessage(role=MessageRole.USER, content="MESSAGE")]
        memories = [MemoryFact(content="MEMORY", user_id=123)]

        result = builder.assemble(
            summary=summary,
            recent_messages=messages,
            semantic_memories=memories,
        )

        # Check order: summary, then memory, then message
        summary_pos = result.find("SUMMARY")
        memory_pos = result.find("MEMORY")
        message_pos = result.find("MESSAGE")

        assert summary_pos < memory_pos < message_pos

    def test_assemble_respects_token_limit(self):
        """Test that assemble respects total token limit."""
        config = ContextAssemblyConfig(
            max_total_tokens=20,
            tokens_per_char=1.0,  # Easy calculation
        )
        builder = ContextBuilder(config)
        summary = RunningSummary(content="a" * 30)  # 30 tokens
        messages = [ConversationMessage(role=MessageRole.USER, content="b" * 30)]

        result = builder.assemble(summary=summary, recent_messages=messages)

        # Should be truncated to fit within 20 tokens
        assert len(result) <= 20


class TestAssembleForLLM:
    """Tests for assemble_for_llm method."""

    def test_assemble_with_system_prompt(self):
        """Test assembling with system prompt."""
        builder = ContextBuilder()

        messages = builder.assemble_for_llm(
            system_prompt="You are a helpful assistant",
        )

        assert len(messages) >= 1
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]

    def test_assemble_includes_context(self):
        """Test that assembled context is included."""
        builder = ContextBuilder()
        summary = RunningSummary(content="User context")

        messages = builder.assemble_for_llm(summary=summary)

        # Should have system prompt for context
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) >= 1
        assert any("User context" in m["content"] for m in system_messages)

    def test_assemble_includes_recent_messages(self):
        """Test that recent messages are included."""
        builder = ContextBuilder()
        recent = [
            ConversationMessage(role=MessageRole.USER, content="Hello"),
            ConversationMessage(role=MessageRole.ASSISTANT, content="Hi!"),
        ]

        messages = builder.assemble_for_llm(recent_messages=recent)

        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]

        assert len(user_msgs) >= 1
        assert len(assistant_msgs) >= 1

    def test_assemble_message_ordering(self):
        """Test that messages are in correct order."""
        builder = ContextBuilder()
        recent = [
            ConversationMessage(role=MessageRole.USER, content="First"),
            ConversationMessage(role=MessageRole.ASSISTANT, content="Second"),
        ]

        messages = builder.assemble_for_llm(
            system_prompt="System",
            recent_messages=recent,
        )

        # Filter out system messages
        non_system = [m for m in messages if m["role"] != "system"]

        assert non_system[0]["content"] == "First"
        assert non_system[1]["content"] == "Second"


class TestGlobalInstance:
    """Tests for global ContextBuilder instance management."""

    def test_get_context_builder_creates_instance(self):
        """Test that get_context_builder creates default instance."""
        set_context_builder(None)  # Reset
        builder = get_context_builder()

        assert isinstance(builder, ContextBuilder)

    def test_get_context_builder_with_config(self):
        """Test that get_context_builder uses provided config."""
        set_context_builder(None)  # Reset
        config = ContextAssemblyConfig(max_total_tokens=500)
        builder = get_context_builder(config)

        assert builder.config.max_total_tokens == 500

    def test_set_context_builder(self):
        """Test setting global instance."""
        custom_builder = ContextBuilder(ContextAssemblyConfig(max_total_tokens=999))
        set_context_builder(custom_builder)

        retrieved = get_context_builder()
        assert retrieved is custom_builder
        assert retrieved.config.max_total_tokens == 999

    def teardown_method(self):
        """Clean up after each test."""
        set_context_builder(None)


class TestArtifactSurrogateConfig:
    """Tests for artifact surrogate configuration."""

    def test_default_artifact_config(self):
        """Test default artifact configuration values."""
        config = ContextAssemblyConfig()

        assert config.max_artifact_surrogates == 5
        assert config.max_artifact_tokens == 800
        assert config.max_surrogates_per_artifact == 2

    def test_custom_artifact_config(self):
        """Test custom artifact configuration."""
        config = ContextAssemblyConfig(
            max_artifact_surrogates=10,
            max_artifact_tokens=1000,
            max_surrogates_per_artifact=3,
        )

        assert config.max_artifact_surrogates == 10
        assert config.max_artifact_tokens == 1000
        assert config.max_surrogates_per_artifact == 3

    def test_default_order_includes_artifacts(self):
        """Test that default order includes artifact_surrogates."""
        config = ContextAssemblyConfig()

        assert "artifact_surrogates" in config.order


class TestBuildArtifactSurrogatesPart:
    """Tests for building artifact surrogates context part."""

    @pytest.fixture
    def sample_surrogates(self):
        """Create sample surrogates for testing."""
        from bot.services.artifact_service import TextSurrogateForContext, ArtifactType, TextSurrogateKind
        return [
            TextSurrogateForContext(
                artifact_id="artifact-1",
                artifact_type=ArtifactType.IMAGE,
                original_filename="photo.jpg",
                text_kind=TextSurrogateKind.VISION_SUMMARY,
                text_content="A photo of a cat",
            ),
            TextSurrogateForContext(
                artifact_id="artifact-2",
                artifact_type=ArtifactType.DOCUMENT,
                original_filename="doc.pdf",
                text_kind=TextSurrogateKind.FILE_SUMMARY,
                text_content="Document about cats",
            ),
        ]

    def test_build_with_surrogates(self, sample_surrogates):
        """Test building part with surrogates."""
        builder = ContextBuilder()

        part = builder.build_artifact_surrogates_part(sample_surrogates)

        assert part is not None
        assert part.source == "artifact_surrogates"
        assert "Attached files:" in part.content
        assert "photo.jpg" in part.content
        assert "doc.pdf" in part.content

    def test_build_with_empty_surrogates(self):
        """Test building part with empty surrogates."""
        builder = ContextBuilder()

        part = builder.build_artifact_surrogates_part([])

        assert part is None

    def test_build_with_none_surrogates(self):
        """Test building part with None surrogates."""
        builder = ContextBuilder()

        part = builder.build_artifact_surrogates_part(None)

        assert part is None

    def test_build_with_dict_surrogates(self):
        """Test building part with dict surrogates."""
        builder = ContextBuilder()
        surrogates = [
            {
                "artifact_id": "artifact-1",
                "artifact_type": "image",
                "original_filename": "photo.jpg",
                "text_kind": "vision_summary",
                "text_content": "A photo of a cat",
            }
        ]

        part = builder.build_artifact_surrogates_part(surrogates)

        assert part is not None
        assert "[image: photo.jpg]" in part.content

    def test_build_truncates_long_content(self):
        """Test that long surrogate content is truncated."""
        from bot.services.artifact_service import TextSurrogateForContext, ArtifactType, TextSurrogateKind
        config = ContextAssemblyConfig(
            max_artifact_tokens=50,
            tokens_per_char=1.0,  # Easy calculation
        )
        builder = ContextBuilder(config)

        surrogates = [
            TextSurrogateForContext(
                artifact_id="artifact-1",
                artifact_type=ArtifactType.IMAGE,
                original_filename="photo.jpg",
                text_kind=TextSurrogateKind.VISION_SUMMARY,
                text_content="A" * 100,  # 100 chars = 100 tokens
            ),
        ]

        part = builder.build_artifact_surrogates_part(surrogates)

        assert part is not None
        # Should be truncated
        assert len(part.content) < 150
        assert part.token_estimate <= config.max_artifact_tokens

    def test_build_prioritizes_surrogates(self):
        """Test that surrogates are included in priority order."""
        from bot.services.artifact_service import TextSurrogateForContext, ArtifactType, TextSurrogateKind
        builder = ContextBuilder()

        surrogates = [
            TextSurrogateForContext(
                artifact_id="artifact-1",
                artifact_type=ArtifactType.IMAGE,
                original_filename="first.jpg",
                text_kind=TextSurrogateKind.VISION_SUMMARY,
                text_content="First image",
            ),
            TextSurrogateForContext(
                artifact_id="artifact-2",
                artifact_type=ArtifactType.IMAGE,
                original_filename="second.jpg",
                text_kind=TextSurrogateKind.VISION_SUMMARY,
                text_content="Second image",
            ),
        ]

        part = builder.build_artifact_surrogates_part(surrogates)

        assert part is not None
        assert "first.jpg" in part.content
        assert "second.jpg" in part.content


class TestAssembleWithArtifactSurrogates:
    """Tests for assemble with artifact surrogates."""

    def test_assemble_includes_surrogates(self):
        """Test that assemble includes artifact surrogates."""
        from bot.services.artifact_service import TextSurrogateForContext, ArtifactType, TextSurrogateKind
        builder = ContextBuilder()

        summary = RunningSummary(content="Summary here")
        surrogates = [
            TextSurrogateForContext(
                artifact_id="artifact-1",
                artifact_type=ArtifactType.IMAGE,
                original_filename="photo.jpg",
                text_kind=TextSurrogateKind.VISION_SUMMARY,
                text_content="A photo of a cat",
            ),
        ]

        result = builder.assemble(
            summary=summary,
            artifact_surrogates=surrogates,
        )

        assert "Summary here" in result
        assert "A photo of a cat" in result
        assert "Attached files:" in result

    def test_assemble_order_with_surrogates(self):
        """Test that surrogates appear in correct order."""
        from bot.services.artifact_service import TextSurrogateForContext, ArtifactType, TextSurrogateKind
        config = ContextAssemblyConfig(order=["summary", "artifact_surrogates", "semantic_memory"])
        builder = ContextBuilder(config)

        summary = RunningSummary(content="SUMMARY")
        surrogates = [
            TextSurrogateForContext(
                artifact_id="artifact-1",
                artifact_type=ArtifactType.IMAGE,
                original_filename="photo.jpg",
                text_kind=TextSurrogateKind.VISION_SUMMARY,
                text_content="SURROGATE",
            ),
        ]
        memories = [MemoryFact(content="MEMORY", user_id=123)]

        result = builder.assemble(
            summary=summary,
            artifact_surrogates=surrogates,
            semantic_memories=memories,
        )

        # Check order: summary, then surrogates, then memories
        summary_pos = result.find("SUMMARY")
        surrogate_pos = result.find("SURROGATE")
        memory_pos = result.find("MEMORY")

        assert summary_pos < surrogate_pos < memory_pos


class TestAssembleForLLMWithSurrogates:
    """Tests for assemble_for_llm with artifact surrogates."""

    def test_assemble_for_llm_includes_surrogates(self):
        """Test that assemble_for_llm includes artifact surrogates."""
        from bot.services.artifact_service import TextSurrogateForContext, ArtifactType, TextSurrogateKind
        builder = ContextBuilder()

        summary = RunningSummary(content="Summary")
        surrogates = [
            TextSurrogateForContext(
                artifact_id="artifact-1",
                artifact_type=ArtifactType.IMAGE,
                original_filename="photo.jpg",
                text_kind=TextSurrogateKind.VISION_SUMMARY,
                text_content="A photo of a cat",
            ),
        ]

        messages = builder.assemble_for_llm(
            summary=summary,
            artifact_surrogates=surrogates,
        )

        # Find system context message
        context_msgs = [m for m in messages if m["role"] == "system"]
        assert len(context_msgs) >= 1

        # Check that surrogates are in context
        context_text = " ".join(m["content"] for m in context_msgs)
        assert "A photo of a cat" in context_text
        assert "Attached files:" in context_text
