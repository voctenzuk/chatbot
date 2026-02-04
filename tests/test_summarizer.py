"""Tests for Summarizer service."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from bot.services.summarizer import (
    Summarizer,
    SummarizerConfig,
    SummaryKind,
    SummaryResult,
    SummaryJSON,
    FactCandidate,
    ArtifactReference,
    get_summarizer,
    set_summarizer,
)


class TestFactCandidate:
    """Tests for FactCandidate dataclass."""

    def test_create_default(self):
        """Test creating fact with defaults."""
        fact = FactCandidate(text="User likes Python")

        assert fact.text == "User likes Python"
        assert fact.confidence == 0.8
        assert fact.category == "general"

    def test_create_custom(self):
        """Test creating fact with custom values."""
        fact = FactCandidate(
            text="User works at Google",
            confidence=0.95,
            category="fact",
        )

        assert fact.text == "User works at Google"
        assert fact.confidence == 0.95
        assert fact.category == "fact"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        fact = FactCandidate(text="Test fact", confidence=0.9, category="preference")

        result = fact.to_dict()

        assert result == {
            "text": "Test fact",
            "confidence": 0.9,
            "category": "preference",
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"text": "Parsed fact", "confidence": 0.85, "category": "goal"}

        fact = FactCandidate.from_dict(data)

        assert fact.text == "Parsed fact"
        assert fact.confidence == 0.85
        assert fact.category == "goal"

    def test_from_dict_defaults(self):
        """Test from_dict with missing fields uses defaults."""
        data = {"text": "Minimal fact"}

        fact = FactCandidate.from_dict(data)

        assert fact.text == "Minimal fact"
        assert fact.confidence == 0.8
        assert fact.category == "general"


class TestArtifactReference:
    """Tests for ArtifactReference dataclass."""

    def test_create_default(self):
        """Test creating artifact ref with defaults."""
        art = ArtifactReference(artifact_id="art_123")

        assert art.artifact_id == "art_123"
        assert art.note == ""

    def test_create_with_note(self):
        """Test creating artifact ref with note."""
        art = ArtifactReference(artifact_id="art_456", note="Screenshot of error")

        assert art.artifact_id == "art_456"
        assert art.note == "Screenshot of error"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        art = ArtifactReference(artifact_id="id1", note="Test note")

        result = art.to_dict()

        assert result == {"artifact_id": "id1", "note": "Test note"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"artifact_id": "id2", "note": "Parsed note"}

        art = ArtifactReference.from_dict(data)

        assert art.artifact_id == "id2"
        assert art.note == "Parsed note"


class TestSummaryJSON:
    """Tests for SummaryJSON dataclass."""

    def test_create_default(self):
        """Test creating summary with defaults."""
        summary = SummaryJSON()

        assert summary.topic == ""
        assert summary.decisions == []
        assert summary.todos == []
        assert summary.facts_candidates == []
        assert summary.entities == []
        assert summary.artifacts == []
        assert summary.open_questions == []

    def test_create_with_data(self):
        """Test creating summary with data."""
        facts = [FactCandidate(text="User likes cats", confidence=0.9)]
        artifacts = [ArtifactReference(artifact_id="img1", note="Cat photo")]

        summary = SummaryJSON(
            topic="Cat discussion",
            decisions=["Adopt a cat"],
            todos=["Visit shelter"],
            facts_candidates=facts,
            entities=["shelter", "cat"],
            artifacts=artifacts,
            open_questions=["Which breed?"],
        )

        assert summary.topic == "Cat discussion"
        assert summary.decisions == ["Adopt a cat"]
        assert len(summary.facts_candidates) == 1
        assert len(summary.artifacts) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = SummaryJSON(
            topic="Test",
            facts_candidates=[FactCandidate(text="Fact 1")],
            artifacts=[ArtifactReference(artifact_id="a1")],
        )

        result = summary.to_dict()

        assert result["topic"] == "Test"
        assert result["decisions"] == []
        assert len(result["facts_candidates"]) == 1
        assert len(result["artifacts"]) == 1

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "topic": "Parsed Topic",
            "decisions": ["D1"],
            "todos": ["T1"],
            "facts_candidates": [{"text": "F1", "confidence": 0.9}],
            "entities": ["E1"],
            "artifacts": [{"artifact_id": "A1", "note": "N1"}],
            "open_questions": ["Q1"],
        }

        summary = SummaryJSON.from_dict(data)

        assert summary.topic == "Parsed Topic"
        assert summary.decisions == ["D1"]
        assert len(summary.facts_candidates) == 1
        assert isinstance(summary.facts_candidates[0], FactCandidate)
        assert len(summary.artifacts) == 1
        assert isinstance(summary.artifacts[0], ArtifactReference)

    def test_from_dict_string_facts(self):
        """Test from_dict handles string facts (backward compatibility)."""
        data = {
            "topic": "Test",
            "facts_candidates": ["String fact 1", "String fact 2"],
        }

        summary = SummaryJSON.from_dict(data)

        assert len(summary.facts_candidates) == 2
        assert summary.facts_candidates[0].text == "String fact 1"
        assert summary.facts_candidates[1].text == "String fact 2"

    def test_get_high_confidence_facts(self):
        """Test filtering facts by confidence."""
        summary = SummaryJSON(
            facts_candidates=[
                FactCandidate(text="High", confidence=0.9),
                FactCandidate(text="Medium", confidence=0.75),
                FactCandidate(text="Low", confidence=0.5),
            ]
        )

        high_facts = summary.get_high_confidence_facts(min_confidence=0.8)

        assert len(high_facts) == 1
        assert high_facts[0].text == "High"

    def test_get_high_confidence_facts_all(self):
        """Test filtering with low threshold returns all."""
        summary = SummaryJSON(
            facts_candidates=[
                FactCandidate(text="A", confidence=0.9),
                FactCandidate(text="B", confidence=0.8),
            ]
        )

        facts = summary.get_high_confidence_facts(min_confidence=0.5)

        assert len(facts) == 2


class TestSummaryResult:
    """Tests for SummaryResult dataclass."""

    def test_create_running(self):
        """Test creating running summary result."""
        result = SummaryResult(
            kind=SummaryKind.RUNNING,
            summary_text="Running summary text",
            episode_id="ep_123",
            message_count=10,
        )

        assert result.kind == SummaryKind.RUNNING
        assert result.summary_text == "Running summary text"
        assert result.summary_json is None
        assert result.episode_id == "ep_123"
        assert result.message_count == 10

    def test_create_final(self):
        """Test creating final summary result."""
        json_data = SummaryJSON(topic="Final Topic")
        result = SummaryResult(
            kind=SummaryKind.FINAL,
            summary_text="Final summary",
            summary_json=json_data,
            episode_id="ep_456",
            message_count=50,
        )

        assert result.kind == SummaryKind.FINAL
        assert result.summary_json is not None
        assert result.summary_json.topic == "Final Topic"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SummaryResult(
            kind=SummaryKind.CHUNK,
            summary_text="Chunk summary",
            message_count=30,
        )

        data = result.to_dict()

        assert data["kind"] == "chunk"
        assert data["summary_text"] == "Chunk summary"
        assert data["message_count"] == 30
        assert data["summary_json"] is None


class TestSummarizerConfig:
    """Tests for SummarizerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SummarizerConfig()

        assert config.running_summary_interval == 10
        assert config.running_summary_max_tokens == 300
        assert config.chunk_summary_interval == 30
        assert config.final_summary_temperature == 0.2
        assert config.min_fact_confidence == 0.7
        assert config.max_facts_per_summary == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = SummarizerConfig(
            running_summary_interval=5,
            min_fact_confidence=0.8,
        )

        assert config.running_summary_interval == 5
        assert config.min_fact_confidence == 0.8


class TestSummarizerInit:
    """Tests for Summarizer initialization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        summarizer = Summarizer()

        assert summarizer.config is not None
        assert summarizer.llm_provider is not None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = SummarizerConfig(running_summary_interval=5)
        summarizer = Summarizer(config=config)

        assert summarizer.config.running_summary_interval == 5

    def test_init_with_llm_provider(self):
        """Test initialization with custom LLM provider."""
        mock_provider = MagicMock()
        summarizer = Summarizer(llm_provider=mock_provider)

        assert summarizer.llm_provider is mock_provider


class TestSummarizerHelpers:
    """Tests for Summarizer helper methods."""

    def test_format_messages(self):
        """Test message formatting."""
        summarizer = Summarizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = summarizer._format_messages(messages)

        assert "User: Hello" in result
        assert "Assistant: Hi!" in result

    def test_format_messages_unknown_role(self):
        """Test formatting with unknown role."""
        summarizer = Summarizer()
        messages = [{"role": "unknown", "content": "Test"}]

        result = summarizer._format_messages(messages)

        assert "Unknown: Test" in result

    def test_truncate_text_under_limit(self):
        """Test truncation when text is under limit."""
        summarizer = Summarizer()
        text = "Short text"

        result = summarizer._truncate_text(text, max_tokens=100)

        assert result == "Short text"

    def test_truncate_text_over_limit(self):
        """Test truncation when text exceeds limit."""
        summarizer = Summarizer()
        text = "A" * 1000

        result = summarizer._truncate_text(text, max_tokens=10)

        assert len(result) < 1000
        assert result.endswith("...")

    def test_fallback_summary(self):
        """Test fallback summary generation."""
        summarizer = Summarizer()
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
        ]

        result = summarizer._fallback_summary(messages, None)

        assert "2 messages" in result
        assert "First message" in result

    def test_fallback_summary_with_previous(self):
        """Test fallback summary with previous context."""
        summarizer = Summarizer()
        messages = [{"role": "user", "content": "New"}]

        result = summarizer._fallback_summary(messages, "Previous summary")

        assert "Previous context" in result


class TestParseSummaryJSON:
    """Tests for JSON parsing."""

    def test_parse_plain_json(self):
        """Test parsing plain JSON."""
        summarizer = Summarizer()
        json_str = json.dumps(
            {
                "topic": "Test",
                "decisions": ["D1"],
                "todos": [],
                "facts_candidates": [],
                "entities": [],
                "artifacts": [],
                "open_questions": [],
            }
        )

        result = summarizer._parse_summary_json(json_str)

        assert result.topic == "Test"

    def test_parse_markdown_json(self):
        """Test parsing JSON in markdown code block."""
        summarizer = Summarizer()
        response = '```json\n{"topic": "Markdown", "decisions": []}\n```'

        result = summarizer._parse_summary_json(response)

        assert result.topic == "Markdown"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns error structure."""
        summarizer = Summarizer()

        result = summarizer._parse_summary_json("not json")

        assert result.topic == "Parse error"
        assert len(result.facts_candidates) == 1
        assert result.facts_candidates[0].confidence == 0.0


class TestCreateTextFromJSON:
    """Tests for text summary creation from JSON."""

    def test_create_with_all_fields(self):
        """Test creating text from full JSON."""
        summarizer = Summarizer()
        json_data = SummaryJSON(
            topic="Discussion",
            decisions=["Decided A", "Decided B"],
            todos=["Todo 1"],
            open_questions=["Question?"],
        )

        result = summarizer._create_text_from_json(json_data)

        assert "Topic: Discussion" in result
        assert "Decided A" in result
        assert "Todo 1" in result
        assert "Question?" in result

    def test_create_empty(self):
        """Test creating text from empty JSON."""
        summarizer = Summarizer()
        json_data = SummaryJSON()

        result = summarizer._create_text_from_json(json_data)

        assert result == "Episode summary unavailable"


class TestTriggerChecks:
    """Tests for summary trigger conditions."""

    def test_should_generate_running_summary_true(self):
        """Test running summary trigger when threshold met."""
        config = SummarizerConfig(running_summary_interval=10)
        summarizer = Summarizer(config=config)

        result = summarizer.should_generate_running_summary(
            message_count=25,
            last_summary_message_count=15,
        )

        assert result is True

    def test_should_generate_running_summary_false(self):
        """Test running summary trigger when below threshold."""
        config = SummarizerConfig(running_summary_interval=10)
        summarizer = Summarizer(config=config)

        result = summarizer.should_generate_running_summary(
            message_count=20,
            last_summary_message_count=15,
        )

        assert result is False

    def test_should_generate_chunk_summary_true(self):
        """Test chunk summary trigger when threshold met."""
        config = SummarizerConfig(chunk_summary_interval=30)
        summarizer = Summarizer(config=config)

        result = summarizer.should_generate_chunk_summary(
            message_count=70,
            last_chunk_message_count=40,
        )

        assert result is True


class TestExtractFactsForMem0:
    """Tests for mem0 fact extraction."""

    def test_extract_high_confidence(self):
        """Test extracting high confidence facts."""
        config = SummarizerConfig(min_fact_confidence=0.8)
        summarizer = Summarizer(config=config)

        summary_json = SummaryJSON(
            facts_candidates=[
                FactCandidate(text="High", confidence=0.9),
                FactCandidate(text="Medium", confidence=0.75),
                FactCandidate(text="Low", confidence=0.5),
            ]
        )
        result = SummaryResult(
            kind=SummaryKind.FINAL,
            summary_text="Test",
            summary_json=summary_json,
        )

        facts = summarizer.extract_facts_for_mem0(result)

        assert len(facts) == 1
        assert facts[0].text == "High"

    def test_extract_no_json(self):
        """Test extraction when no JSON present."""
        summarizer = Summarizer()
        result = SummaryResult(
            kind=SummaryKind.RUNNING,
            summary_text="No JSON",
            summary_json=None,
        )

        facts = summarizer.extract_facts_for_mem0(result)

        assert facts == []

    def test_extract_override_confidence(self):
        """Test extraction with override confidence."""
        summarizer = Summarizer()

        summary_json = SummaryJSON(
            facts_candidates=[
                FactCandidate(text="A", confidence=0.6),
                FactCandidate(text="B", confidence=0.7),
            ]
        )
        result = SummaryResult(
            kind=SummaryKind.FINAL,
            summary_text="Test",
            summary_json=summary_json,
        )

        facts = summarizer.extract_facts_for_mem0(result, min_confidence=0.65)

        assert len(facts) == 1
        assert facts[0].text == "B"

    def test_extract_explicit_zero_confidence(self):
        """Test that explicit min_confidence=0.0 is honored (not treated as falsy)."""
        config = SummarizerConfig(min_fact_confidence=0.8)
        summarizer = Summarizer(config=config)

        summary_json = SummaryJSON(
            facts_candidates=[
                FactCandidate(text="Low confidence", confidence=0.1),
                FactCandidate(text="Medium", confidence=0.5),
                FactCandidate(text="High", confidence=0.9),
            ]
        )
        result = SummaryResult(
            kind=SummaryKind.FINAL,
            summary_text="Test",
            summary_json=summary_json,
        )

        # Explicit 0.0 should return ALL facts, not use config default (0.8)
        facts = summarizer.extract_facts_for_mem0(result, min_confidence=0.0)

        assert len(facts) == 3
        assert facts[0].text == "Low confidence"
        assert facts[1].text == "Medium"
        assert facts[2].text == "High"

    def test_extract_max_facts_per_summary(self):
        """Test that max_facts_per_summary is enforced."""
        config = SummarizerConfig(min_fact_confidence=0.0, max_facts_per_summary=3)
        summarizer = Summarizer(config=config)

        summary_json = SummaryJSON(
            facts_candidates=[
                FactCandidate(text="Fact 1", confidence=0.9),
                FactCandidate(text="Fact 2", confidence=0.85),
                FactCandidate(text="Fact 3", confidence=0.8),
                FactCandidate(text="Fact 4", confidence=0.75),
                FactCandidate(text="Fact 5", confidence=0.7),
            ]
        )
        result = SummaryResult(
            kind=SummaryKind.FINAL,
            summary_text="Test",
            summary_json=summary_json,
        )

        facts = summarizer.extract_facts_for_mem0(result)

        # Should be limited to max_facts_per_summary (3)
        assert len(facts) == 3
        # Should keep order (first 3 match the threshold)
        assert facts[0].text == "Fact 1"
        assert facts[1].text == "Fact 2"
        assert facts[2].text == "Fact 3"


@pytest.mark.asyncio
class TestGenerateRunningSummary:
    """Tests for running summary generation."""

    async def test_generate_running_success(self):
        """Test successful running summary generation."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Generated summary")

        summarizer = Summarizer(llm_provider=mock_llm)
        messages = [{"role": "user", "content": "Hello"}]

        result = await summarizer.generate_running_summary(
            messages=messages,
            previous_summary=None,
            episode_id="ep_1",
        )

        assert result.kind == SummaryKind.RUNNING
        assert result.summary_text == "Generated summary"
        assert result.episode_id == "ep_1"
        assert result.message_count == 1
        mock_llm.generate.assert_called_once()

    async def test_generate_running_with_previous(self):
        """Test running summary with previous context."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Updated summary")

        summarizer = Summarizer(llm_provider=mock_llm)
        messages = [{"role": "user", "content": "New message"}]

        result = await summarizer.generate_running_summary(
            messages=messages,
            previous_summary="Previous context",
            episode_id="ep_1",
        )

        assert result.summary_text == "Updated summary"
        # Verify prompt includes previous summary
        prompt = mock_llm.generate.call_args[0][0]
        assert "Previous context" in prompt

    async def test_generate_running_fallback_on_error(self):
        """Test fallback when LLM fails."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        summarizer = Summarizer(llm_provider=mock_llm)
        messages = [{"role": "user", "content": "Test"}]

        result = await summarizer.generate_running_summary(messages=messages)

        assert result.kind == SummaryKind.RUNNING
        assert "1 messages" in result.summary_text  # Fallback format


@pytest.mark.asyncio
class TestGenerateFinalSummary:
    """Tests for final summary generation."""

    async def test_generate_final_success(self):
        """Test successful final summary generation."""
        json_response = json.dumps(
            {
                "topic": "Final Topic",
                "decisions": ["Decision 1"],
                "todos": ["Todo 1"],
                "facts_candidates": [{"text": "Fact", "confidence": 0.9}],
                "entities": ["Entity"],
                "artifacts": [],
                "open_questions": ["Question?"],
            }
        )
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=json_response)

        summarizer = Summarizer(llm_provider=mock_llm)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = await summarizer.generate_final_summary(
            messages=messages,
            episode_id="ep_1",
        )

        assert result.kind == SummaryKind.FINAL
        assert result.summary_json is not None
        assert result.summary_json.topic == "Final Topic"
        assert len(result.summary_json.facts_candidates) == 1
        assert result.episode_id == "ep_1"
        assert result.message_count == 2

    async def test_generate_final_with_previous_summaries(self):
        """Test final summary with previous summaries."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value='{"topic": "T"}')

        summarizer = Summarizer(llm_provider=mock_llm)
        messages = [{"role": "user", "content": "Test"}]

        await summarizer.generate_final_summary(
            messages=messages,
            previous_summaries=["Summary 1", "Summary 2"],
        )

        prompt = mock_llm.generate.call_args[0][0]
        assert "Previous summaries" in prompt
        assert "Summary 1" in prompt

    async def test_generate_final_fallback_on_error(self):
        """Test fallback when LLM fails."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        summarizer = Summarizer(llm_provider=mock_llm)
        messages = [{"role": "user", "content": "Test"}]

        result = await summarizer.generate_final_summary(messages=messages)

        assert result.kind == SummaryKind.FINAL
        assert result.summary_json is not None
        assert result.summary_json.topic == "Unknown"

    async def test_generate_final_uses_low_temperature(self):
        """Test that final summary uses low temperature."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value='{"topic": "T"}')

        config = SummarizerConfig(final_summary_temperature=0.2)
        summarizer = Summarizer(config=config, llm_provider=mock_llm)
        messages = [{"role": "user", "content": "Test"}]

        await summarizer.generate_final_summary(messages=messages)

        temp = mock_llm.generate.call_args[1]["temperature"]
        assert temp == 0.2


@pytest.mark.asyncio
class TestGenerateChunkSummary:
    """Tests for chunk summary generation."""

    async def test_generate_chunk_success(self):
        """Test successful chunk summary generation."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Chunk summary")

        summarizer = Summarizer(llm_provider=mock_llm)
        messages = [{"role": "user", "content": "M" * 50}]

        result = await summarizer.generate_chunk_summary(
            messages=messages,
            episode_id="ep_1",
        )

        assert result.kind == SummaryKind.CHUNK
        assert result.summary_text == "Chunk summary"
        assert result.episode_id == "ep_1"

    async def test_generate_chunk_fallback_on_error(self):
        """Test fallback when LLM fails."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        summarizer = Summarizer(llm_provider=mock_llm)
        messages = [{"role": "user", "content": "Test"}]

        result = await summarizer.generate_chunk_summary(messages=messages)

        assert result.kind == SummaryKind.CHUNK
        assert "1 messages" in result.summary_text  # Fallback format


class TestGlobalInstance:
    """Tests for global Summarizer instance management."""

    def test_get_summarizer_creates_instance(self):
        """Test that get_summarizer creates default instance."""
        set_summarizer(None)  # Reset
        summarizer = get_summarizer()

        assert isinstance(summarizer, Summarizer)

    def test_get_summarizer_with_config(self):
        """Test that get_summarizer uses provided config."""
        set_summarizer(None)  # Reset
        config = SummarizerConfig(running_summary_interval=5)
        summarizer = get_summarizer(config)

        assert summarizer.config.running_summary_interval == 5

    def test_get_summarizer_honors_llm_provider_override(self):
        """Test that get_summarizer rebuilds when llm_provider differs."""
        set_summarizer(None)  # Reset

        # Create first instance with mock provider
        mock_provider_1 = MagicMock()
        summarizer_1 = get_summarizer(llm_provider=mock_provider_1)
        assert summarizer_1.llm_provider is mock_provider_1

        # Calling again with same provider should return same instance
        summarizer_1_again = get_summarizer(llm_provider=mock_provider_1)
        assert summarizer_1_again is summarizer_1

        # Calling with different provider should create new instance
        mock_provider_2 = MagicMock()
        summarizer_2 = get_summarizer(llm_provider=mock_provider_2)
        assert summarizer_2 is not summarizer_1
        assert summarizer_2.llm_provider is mock_provider_2

    def test_get_summarizer_no_rebuild_when_provider_same(self):
        """Test that get_summarizer does not rebuild when same provider passed."""
        set_summarizer(None)  # Reset

        mock_provider = MagicMock()
        summarizer_1 = get_summarizer(llm_provider=mock_provider)
        summarizer_2 = get_summarizer(llm_provider=mock_provider)

        assert summarizer_1 is summarizer_2

    def test_set_summarizer(self):
        """Test setting global instance."""
        custom = Summarizer(SummarizerConfig(running_summary_interval=99))
        set_summarizer(custom)

        retrieved = get_summarizer()
        assert retrieved is custom
        assert retrieved.config.running_summary_interval == 99

    def teardown_method(self):
        """Clean up after each test."""
        set_summarizer(None)
