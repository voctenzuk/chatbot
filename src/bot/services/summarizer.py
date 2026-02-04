"""Summarizer service for episode summaries.

This module provides the Summarizer class that generates:
- Running summaries (incremental updates during conversations)
- Final episode summaries (structured JSON format with facts_candidates)

The service extracts facts_candidates for mem0 integration as specified in
ARCHITECTURE/MEMORY_DESIGN.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from loguru import logger


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        ...


class SimpleLLMProvider:
    """Simple LLM provider using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._client = None

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using OpenAI API."""
        if self._client is None:
            try:
                import openai

                self._client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise RuntimeError("openai package is required for SimpleLLMProvider")

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class SummaryKind(Enum):
    """Type of summary."""

    RUNNING = "running"  # Incremental summary during conversation
    CHUNK = "chunk"  # Periodic summary for long episodes
    FINAL = "final"  # Final summary when episode closes


@dataclass
class FactCandidate:
    """A candidate fact for mem0 extraction.

    Represents a potential memory unit that could be stored in mem0.
    Includes confidence score for filtering.
    """

    text: str
    confidence: float = 0.8
    category: str = "general"  # e.g., "preference", "fact", "goal"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactCandidate:
        """Create from dictionary."""
        return cls(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.8),
            category=data.get("category", "general"),
        )


@dataclass
class ArtifactReference:
    """Reference to an artifact (attachment) in the conversation."""

    artifact_id: str
    note: str = ""  # Description of how the artifact relates to the episode

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactReference:
        """Create from dictionary."""
        return cls(
            artifact_id=data.get("artifact_id", ""),
            note=data.get("note", ""),
        )


@dataclass
class SummaryJSON:
    """Structured summary format per MEMORY_DESIGN.md.

    This is the JSON structure stored in episode_summaries.summary_json.
    """

    topic: str = ""  # Main topic of the episode
    decisions: list[str] = field(default_factory=list)  # Decisions made
    todos: list[str] = field(default_factory=list)  # Open todos/goals
    facts_candidates: list[FactCandidate] = field(default_factory=list)  # For mem0
    entities: list[str] = field(default_factory=list)  # People, places, things
    artifacts: list[ArtifactReference] = field(default_factory=list)  # Attachments
    open_questions: list[str] = field(default_factory=list)  # Unresolved questions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topic": self.topic,
            "decisions": self.decisions,
            "todos": self.todos,
            "facts_candidates": [f.to_dict() for f in self.facts_candidates],
            "entities": self.entities,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "open_questions": self.open_questions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SummaryJSON:
        """Create from dictionary."""
        facts = [
            FactCandidate.from_dict(f) if isinstance(f, dict) else FactCandidate(text=str(f))
            for f in data.get("facts_candidates", [])
        ]
        artifacts = [
            ArtifactReference.from_dict(a)
            if isinstance(a, dict)
            else ArtifactReference(artifact_id=str(a))
            for a in data.get("artifacts", [])
        ]
        return cls(
            topic=data.get("topic", ""),
            decisions=data.get("decisions", []),
            todos=data.get("todos", []),
            facts_candidates=facts,
            entities=data.get("entities", []),
            artifacts=artifacts,
            open_questions=data.get("open_questions", []),
        )

    def get_high_confidence_facts(self, min_confidence: float = 0.7) -> list[FactCandidate]:
        """Get facts with confidence above threshold.

        Args:
            min_confidence: Minimum confidence score (0.0-1.0)

        Returns:
            List of facts meeting the confidence threshold
        """
        return [f for f in self.facts_candidates if f.confidence >= min_confidence]


@dataclass
class SummaryResult:
    """Result of a summarization operation."""

    kind: SummaryKind
    summary_text: str  # Short, LLM-friendly text summary
    summary_json: SummaryJSON | None = None  # Structured data (for final summaries)
    episode_id: str | None = None
    message_count: int = 0  # Number of messages summarized
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kind": self.kind.value,
            "summary_text": self.summary_text,
            "summary_json": self.summary_json.to_dict() if self.summary_json else None,
            "episode_id": self.episode_id,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SummarizerConfig:
    """Configuration for summarizer behavior."""

    # Running summary triggers
    running_summary_interval: int = 10  # Messages between running summaries
    running_summary_max_tokens: int = 300  # Max tokens for running summary

    # Chunk summary triggers (for very long episodes)
    chunk_summary_interval: int = 30  # Messages between chunk summaries
    chunk_summary_max_tokens: int = 500

    # Final summary
    final_summary_temperature: float = 0.2  # Low temp for consistent extraction
    final_summary_max_tokens: int = 800

    # Fact extraction
    min_fact_confidence: float = 0.7  # Minimum confidence for facts to store
    max_facts_per_summary: int = 10  # Limit facts extracted

    # Prompt templates (can be customized)
    running_summary_prompt_template: str = (
        "Summarize the following conversation concisely (1-2 paragraphs). "
        "Focus on the current topic, key points discussed, and any open questions or goals.\n\n"
        "Previous summary:\n{previous_summary}\n\n"
        "New messages:\n{messages}\n\n"
        "Updated summary:"
    )

    final_summary_prompt_template: str = (
        "Analyze this conversation and provide a structured summary. "
        "Return ONLY a JSON object with no markdown formatting.\n\n"
        "Conversation:\n<<MESSAGES_PLACEHOLDER>>\n\n"
        "JSON format (all fields required):\n"
        "{\n"
        '  "topic": "main topic of conversation",\n'
        '  "decisions": ["decision 1", "decision 2"],\n'
        '  "todos": ["todo 1", "todo 2"],\n'
        '  "facts_candidates": [\n'
        '    {"text": "fact about user", "confidence": 0.9, "category": "preference"}\n'
        "  ],\n"
        '  "entities": ["person1", "place1"],\n'
        '  "artifacts": [{"artifact_id": "id1", "note": "description"}],\n'
        '  "open_questions": ["question 1"]\n'
        "}\n\n"
        "Facts should include user preferences, important information, and decisions. "
        "Confidence should be 0.0-1.0 based on clarity and importance. "
        "Categories: preference, fact, goal, relationship, general."
    )


class Summarizer:
    """Summarizer service for episode management.

    Features:
    - Running summaries: Incremental updates during active conversations
    - Chunk summaries: Periodic summaries for long episodes
    - Final summaries: Structured JSON output with facts_candidates for mem0
    """

    def __init__(
        self,
        config: SummarizerConfig | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize summarizer.

        Args:
            config: Configuration for summarizer behavior
            llm_provider: LLM provider for generating summaries
        """
        self.config = config or SummarizerConfig()
        self.llm_provider = llm_provider or SimpleLLMProvider()

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for prompt.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted message string
        """
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            role_label = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
                role, role.capitalize()
            )
            lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit.

        Uses a simple character-based heuristic (4 chars ≈ 1 token).

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    async def generate_running_summary(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str | None = None,
        episode_id: str | None = None,
    ) -> SummaryResult:
        """Generate a running summary.

        Creates an incremental summary that incorporates new messages
        into any previous summary.

        Args:
            messages: Recent messages to summarize
            previous_summary: Optional previous running summary to build upon
            episode_id: Optional episode identifier

        Returns:
            SummaryResult with kind=RUNNING
        """
        formatted_messages = self._format_messages(messages)

        prompt = self.config.running_summary_prompt_template.format(
            previous_summary=previous_summary or "(No previous summary)",
            messages=formatted_messages,
        )

        prompt = self._truncate_text(prompt, self.config.running_summary_max_tokens * 2)

        try:
            summary_text = await self.llm_provider.generate(
                prompt,
                temperature=0.5,  # Moderate temp for coherent summaries
            )
            summary_text = self._truncate_text(
                summary_text.strip(), self.config.running_summary_max_tokens
            )
        except Exception as e:
            logger.error("Failed to generate running summary: {}", e)
            # Fallback to simple concatenation
            summary_text = self._fallback_summary(messages, previous_summary)

        return SummaryResult(
            kind=SummaryKind.RUNNING,
            summary_text=summary_text,
            episode_id=episode_id,
            message_count=len(messages),
        )

    async def generate_final_summary(
        self,
        messages: list[dict[str, Any]],
        episode_id: str | None = None,
        previous_summaries: list[str] | None = None,
    ) -> SummaryResult:
        """Generate a final episode summary with structured JSON.

        Creates a comprehensive summary with facts_candidates for mem0 extraction.

        Args:
            messages: All messages from the episode
            episode_id: Optional episode identifier
            previous_summaries: Optional list of previous running summaries

        Returns:
            SummaryResult with kind=FINAL and structured summary_json
        """
        formatted_messages = self._format_messages(messages)

        # Include previous summaries for context if available
        context = ""
        if previous_summaries:
            context = "Previous summaries:\n" + "\n".join(previous_summaries) + "\n\n"

        prompt = self.config.final_summary_prompt_template.replace(
            "<<MESSAGES_PLACEHOLDER>>",
            context + "Full conversation:\n" + formatted_messages,
        )

        try:
            response = await self.llm_provider.generate(
                prompt, temperature=self.config.final_summary_temperature
            )
            response = response.strip()

            # Parse JSON response
            summary_json = self._parse_summary_json(response)
            summary_text = self._create_text_from_json(summary_json)

        except Exception as e:
            logger.error("Failed to generate final summary: {}", e)
            # Fallback
            summary_text = self._fallback_summary(messages, None)
            summary_json = SummaryJSON(
                topic="Unknown",
                facts_candidates=[],
            )

        return SummaryResult(
            kind=SummaryKind.FINAL,
            summary_text=summary_text,
            summary_json=summary_json,
            episode_id=episode_id,
            message_count=len(messages),
        )

    async def generate_chunk_summary(
        self,
        messages: list[dict[str, Any]],
        episode_id: str | None = None,
    ) -> SummaryResult:
        """Generate a chunk summary for long episodes.

        Similar to running summary but for periodic checkpoints.

        Args:
            messages: Messages in this chunk
            episode_id: Optional episode identifier

        Returns:
            SummaryResult with kind=CHUNK
        """
        formatted_messages = self._format_messages(messages)

        prompt = (
            "Summarize this section of a longer conversation (1 paragraph). "
            "Focus on key points and decisions.\n\n"
            f"{formatted_messages}\n\n"
            "Summary:"
        )

        prompt = self._truncate_text(prompt, self.config.chunk_summary_max_tokens * 2)

        try:
            summary_text = await self.llm_provider.generate(prompt, temperature=0.5)
            summary_text = self._truncate_text(
                summary_text.strip(), self.config.chunk_summary_max_tokens
            )
        except Exception as e:
            logger.error("Failed to generate chunk summary: {}", e)
            summary_text = self._fallback_summary(messages, None)

        return SummaryResult(
            kind=SummaryKind.CHUNK,
            summary_text=summary_text,
            episode_id=episode_id,
            message_count=len(messages),
        )

    def _parse_summary_json(self, response: str) -> SummaryJSON:
        """Parse JSON response from LLM.

        Handles various JSON formats (with/without markdown code blocks).

        Args:
            response: LLM response string

        Returns:
            Parsed SummaryJSON
        """
        # Try to extract JSON from markdown code blocks
        text = response.strip()

        # Remove markdown code block if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])
            else:
                text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("`") and text.endswith("`"):
            text = text[1:-1]

        try:
            data = json.loads(text)
            return SummaryJSON.from_dict(data)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse summary JSON: {}", e)
            # Try to extract JSON-like content with regex
            import re

            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return SummaryJSON.from_dict(data)
                except json.JSONDecodeError:
                    pass

            # Return empty structure
            return SummaryJSON(
                topic="Parse error",
                facts_candidates=[FactCandidate(text="Summary parsing failed", confidence=0.0)],
            )

    def _create_text_from_json(self, summary_json: SummaryJSON) -> str:
        """Create readable text summary from structured JSON.

        Args:
            summary_json: Structured summary data

        Returns:
            Human-readable summary text
        """
        parts = []

        if summary_json.topic:
            parts.append(f"Topic: {summary_json.topic}")

        if summary_json.decisions:
            parts.append("Decisions: " + "; ".join(summary_json.decisions))

        if summary_json.todos:
            parts.append("Todos: " + "; ".join(summary_json.todos))

        if summary_json.open_questions:
            parts.append("Open questions: " + "; ".join(summary_json.open_questions))

        return "\n".join(parts) if parts else "Episode summary unavailable"

    def _fallback_summary(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str | None,
    ) -> str:
        """Create a simple fallback summary without LLM.

        Args:
            messages: Messages to summarize
            previous_summary: Optional previous summary

        Returns:
            Simple text summary
        """
        message_count = len(messages)
        first_msg = messages[0].get("content", "")[:100] if messages else ""
        last_msg = messages[-1].get("content", "")[:100] if messages else ""

        summary = f"Conversation with {message_count} messages."
        if first_msg:
            summary += f" Started with: {first_msg}..."
        if last_msg and len(messages) > 1:
            summary += f" Latest: {last_msg}..."
        if previous_summary:
            summary = f"Previous context: {previous_summary[:200]}...\n{summary}"

        return summary

    def should_generate_running_summary(
        self, message_count: int, last_summary_message_count: int = 0
    ) -> bool:
        """Check if it's time to generate a running summary.

        Args:
            message_count: Current message count
            last_summary_message_count: Message count at last summary

        Returns:
            True if summary should be generated
        """
        messages_since_last = message_count - last_summary_message_count
        return messages_since_last >= self.config.running_summary_interval

    def should_generate_chunk_summary(
        self, message_count: int, last_chunk_message_count: int = 0
    ) -> bool:
        """Check if it's time to generate a chunk summary.

        Args:
            message_count: Current message count
            last_chunk_message_count: Message count at last chunk summary

        Returns:
            True if chunk summary should be generated
        """
        messages_since_last = message_count - last_chunk_message_count
        return messages_since_last >= self.config.chunk_summary_interval

    def extract_facts_for_mem0(
        self,
        summary_result: SummaryResult,
        min_confidence: float | None = None,
    ) -> list[FactCandidate]:
        """Extract high-confidence facts for mem0 storage.

        Args:
            summary_result: Summary result with facts_candidates
            min_confidence: Optional override for minimum confidence

        Returns:
            List of facts meeting confidence threshold
        """
        if not summary_result.summary_json:
            return []

        threshold = self.config.min_fact_confidence if min_confidence is None else min_confidence
        facts = summary_result.summary_json.get_high_confidence_facts(threshold)
        return facts[: self.config.max_facts_per_summary]


# Global instance for dependency injection
_summarizer: Summarizer | None = None


def get_summarizer(
    config: SummarizerConfig | None = None,
    llm_provider: LLMProvider | None = None,
) -> Summarizer:
    """Get or create global Summarizer instance.

    Args:
        config: Optional configuration
        llm_provider: Optional LLM provider

    Returns:
        Summarizer instance
    """
    global _summarizer
    if _summarizer is None or config is not None:
        _summarizer = Summarizer(config, llm_provider)
    elif llm_provider is not None and llm_provider is not _summarizer.llm_provider:
        _summarizer = Summarizer(config, llm_provider)
    assert _summarizer is not None
    return _summarizer


def set_summarizer(summarizer: Summarizer | None) -> None:
    """Set global Summarizer instance (useful for testing).

    Args:
        summarizer: Summarizer instance or None to reset
    """
    global _summarizer
    _summarizer = summarizer
