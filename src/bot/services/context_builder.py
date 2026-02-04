"""ContextBuilder service for dual-memory prompt assembly.

This module provides the ContextBuilder class that assembles prompts from multiple
memory sources:
- Running summary (long-term conversation summary)
- Last N messages (recent conversation history)
- mem0.search results (top-K semantic memories)
- Pruning based on relevance, size limits, and metadata filters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from bot.services.memory_models import MemoryCategory, MemoryFact, MemoryType


class MessageRole(Enum):
    """Role of a message in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """A single message in conversation history."""

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for LLM APIs."""
        return {
            "role": self.role.value,
            "content": self.content,
        }


@dataclass
class RunningSummary:
    """Running summary of conversation history."""

    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_count: int = 0  # Number of messages summarized
    version: int = 1  # Summary version for tracking updates

    def is_stale(self, max_age_hours: int = 24) -> bool:
        """Check if summary is older than specified hours."""
        age = datetime.now() - self.timestamp
        return age.total_seconds() > max_age_hours * 3600


@dataclass
class ContextPart:
    """A part of the assembled context with metadata."""

    content: str
    source: str  # e.g., "summary", "recent_messages", "semantic_memory"
    priority: int  # Higher = more important
    timestamp: datetime = field(default_factory=datetime.now)
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextAssemblyConfig:
    """Configuration for context assembly."""

    # Size limits
    max_total_tokens: int = 4000
    max_summary_tokens: int = 500
    max_recent_messages: int = 10
    max_recent_messages_tokens: int = 1500
    max_semantic_memories: int = 5
    max_semantic_tokens: int = 1000

    # Artifact surrogate limits
    max_artifact_surrogates: int = 5
    max_artifact_tokens: int = 800
    max_surrogates_per_artifact: int = 2

    # Token estimation (approximate)
    tokens_per_char: float = 0.25

    # Metadata filters
    include_categories: list[MemoryCategory] | None = None
    exclude_categories: list[MemoryCategory] | None = None
    include_types: list[MemoryType] | None = None
    exclude_types: list[MemoryType] | None = None
    min_importance_score: float = 0.0

    # Pruning strategy
    prune_by: str = "priority"  # "priority", "recency", "relevance"

    # Ordering
    order: list[str] = field(
        default_factory=lambda: [
            "summary",
            "artifact_surrogates",
            "semantic_memory",
            "recent_messages",
        ]
    )


class ContextBuilder:
    """Builds context for LLM prompts from multiple memory sources.

    Features:
    - Deterministic ordering of context parts
    - Size limits with configurable token estimation
    - Metadata filters for semantic memories
    - Pruning based on priority, recency, or relevance
    """

    def __init__(self, config: ContextAssemblyConfig | None = None):
        """Initialize ContextBuilder with configuration.

        Args:
            config: Configuration for context assembly. Uses defaults if None.
        """
        self.config = config or ContextAssemblyConfig()
        self._token_estimates: dict[str, int] = {}

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple character-based heuristic. For production,
        consider using a proper tokenizer.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        return int(len(text) * self.config.tokens_per_char)

    def _filter_memories(
        self,
        memories: list[MemoryFact],
    ) -> list[MemoryFact]:
        """Filter memories based on metadata filters.

        Args:
            memories: List of memories to filter.

        Returns:
            Filtered list of memories.
        """
        filtered = memories

        # Filter by category
        if self.config.include_categories:
            filtered = [m for m in filtered if m.memory_category in self.config.include_categories]
        if self.config.exclude_categories:
            filtered = [
                m for m in filtered if m.memory_category not in self.config.exclude_categories
            ]

        # Filter by type
        if self.config.include_types:
            filtered = [m for m in filtered if m.memory_type in self.config.include_types]
        if self.config.exclude_types:
            filtered = [m for m in filtered if m.memory_type not in self.config.exclude_types]

        # Filter by importance score
        filtered = [
            m for m in filtered if m.get_effective_importance() >= self.config.min_importance_score
        ]

        return filtered

    def _sort_memories(self, memories: list[MemoryFact]) -> list[MemoryFact]:
        """Sort memories based on pruning strategy.

        Args:
            memories: List of memories to sort.

        Returns:
            Sorted list of memories.
        """
        if self.config.prune_by == "priority":
            # Sort by importance score (descending)
            return sorted(
                memories,
                key=lambda m: m.get_effective_importance(),
                reverse=True,
            )
        elif self.config.prune_by == "recency":
            # Sort by timestamp (newest first)
            return sorted(
                memories,
                key=lambda m: m.timestamp,
                reverse=True,
            )
        elif self.config.prune_by == "relevance":
            # Sort by access count as proxy for relevance
            return sorted(
                memories,
                key=lambda m: m.access_count,
                reverse=True,
            )
        else:
            # Default: no sorting (preserve input order)
            return memories

    def _prune_to_token_limit(
        self,
        parts: list[ContextPart],
        max_tokens: int,
    ) -> list[ContextPart]:
        """Prune context parts to fit within token limit.

        Args:
            parts: List of context parts.
            max_tokens: Maximum tokens allowed.

        Returns:
            Pruned list of context parts.
        """
        total_tokens = 0
        pruned = []

        for part in parts:
            part_tokens = part.token_estimate or self._estimate_tokens(part.content)
            if total_tokens + part_tokens <= max_tokens:
                pruned.append(part)
                total_tokens += part_tokens
            else:
                break

        return pruned

    def build_summary_part(
        self,
        summary: RunningSummary | None,
    ) -> ContextPart | None:
        """Build context part from running summary.

        Args:
            summary: Running summary or None.

        Returns:
            ContextPart if summary exists and fits limits, None otherwise.
        """
        if not summary or not summary.content:
            return None

        token_estimate = self._estimate_tokens(summary.content)

        # Truncate if exceeds max_summary_tokens
        if token_estimate > self.config.max_summary_tokens:
            # Simple truncation: cut at character limit
            max_chars = int(self.config.max_summary_tokens / self.config.tokens_per_char)
            content = summary.content[:max_chars] + "..."
            token_estimate = self.config.max_summary_tokens
        else:
            content = summary.content

        return ContextPart(
            content=content,
            source="summary",
            priority=10,  # Highest priority
            timestamp=summary.timestamp,
            token_estimate=token_estimate,
            metadata={
                "message_count": summary.message_count,
                "version": summary.version,
            },
        )

    def build_recent_messages_part(
        self,
        messages: list[ConversationMessage],
    ) -> ContextPart | None:
        """Build context part from recent messages.

        Args:
            messages: List of recent conversation messages.

        Returns:
            ContextPart if messages exist, None otherwise.
        """
        if not messages:
            return None

        # Take last N messages
        recent = messages[-self.config.max_recent_messages :]

        # Format messages
        lines = []
        for msg in recent:
            role_label = {
                MessageRole.USER: "User",
                MessageRole.ASSISTANT: "Assistant",
                MessageRole.SYSTEM: "System",
            }.get(msg.role, msg.role.value)
            lines.append(f"{role_label}: {msg.content}")

        content = "\n".join(lines)
        token_estimate = self._estimate_tokens(content)

        # Truncate if exceeds limit
        if token_estimate > self.config.max_recent_messages_tokens:
            max_chars = int(self.config.max_recent_messages_tokens / self.config.tokens_per_char)
            content = content[-max_chars:]  # Keep most recent
            if not content.startswith("User:") and not content.startswith("Assistant:"):
                content = "..." + content
            token_estimate = self.config.max_recent_messages_tokens

        return ContextPart(
            content=content,
            source="recent_messages",
            priority=5,  # Medium priority
            timestamp=recent[-1].timestamp if recent else datetime.now(),
            token_estimate=token_estimate,
            metadata={"message_count": len(recent)},
        )

    def build_semantic_memory_part(
        self,
        memories: list[MemoryFact],
        query: str | None = None,
    ) -> ContextPart | None:
        """Build context part from semantic memories.

        Args:
            memories: List of semantic memories from mem0.search.
            query: Optional search query for context.

        Returns:
            ContextPart if memories exist after filtering, None otherwise.
        """
        if not memories:
            return None

        # Apply metadata filters
        filtered = self._filter_memories(memories)

        # Sort based on strategy
        sorted_memories = self._sort_memories(filtered)

        # Take top-K
        top_memories = sorted_memories[: self.config.max_semantic_memories]

        if not top_memories:
            return None

        # Format memories
        lines = []
        for memory in top_memories:
            prefix = ""
            if memory.importance_score > 1.5:
                prefix = "[Important] "
            elif memory.memory_category == MemoryCategory.EMOTIONAL:
                prefix = "[Emotional] "

            lines.append(f"- {prefix}{memory.content}")

        header = f"Relevant memories about: {query}\n" if query else "Relevant memories:\n"
        content = header + "\n".join(lines)
        token_estimate = self._estimate_tokens(content)

        # Truncate if exceeds limit
        if token_estimate > self.config.max_semantic_tokens:
            max_chars = int(self.config.max_semantic_tokens / self.config.tokens_per_char)
            content = content[:max_chars] + "..."
            token_estimate = self.config.max_semantic_tokens

        return ContextPart(
            content=content,
            source="semantic_memory",
            priority=7,  # High priority
            timestamp=top_memories[0].timestamp if top_memories else datetime.now(),
            token_estimate=token_estimate,
            metadata={
                "memory_count": len(top_memories),
                "query": query,
                "categories": list(set(m.memory_category.value for m in top_memories)),
            },
        )

    def build_artifact_surrogates_part(
        self,
        surrogates: list[Any],
    ) -> ContextPart | None:
        """Build context part from artifact text surrogates.

        Args:
            surrogates: List of text surrogates from artifacts (e.g.,
                       TextSurrogateForContext from ArtifactService).

        Returns:
            ContextPart if surrogates exist, None otherwise.
        """
        if not surrogates:
            return None

        # Format surrogates
        lines = []
        for surrogate in surrogates:
            # Handle TextSurrogateForContext or dict
            if hasattr(surrogate, "to_context_string"):
                lines.append(surrogate.to_context_string())
            elif isinstance(surrogate, dict):
                # Build from dict
                artifact_type = surrogate.get("artifact_type", "file")
                filename = surrogate.get("original_filename", "unnamed")
                content = surrogate.get("text_content", "")
                chunk_info = surrogate.get("chunk_info")

                ref = f"[{artifact_type}: {filename}]"
                if chunk_info:
                    ref += f" ({chunk_info})"
                lines.append(f"{ref} {content}")
            else:
                # Fallback: assume string
                lines.append(str(surrogate))

        header = "Attached files:\n"
        content = header + "\n".join(f"- {line}" for line in lines)
        token_estimate = self._estimate_tokens(content)

        # Truncate if exceeds limit
        if token_estimate > self.config.max_artifact_tokens:
            max_chars = int(self.config.max_artifact_tokens / self.config.tokens_per_char)
            # Try to keep complete surrogate entries
            truncated_lines = []
            current_chars = len(header)
            for line in lines:
                if current_chars + len(line) + 3 <= max_chars:  # +3 for "\n- "
                    truncated_lines.append(line)
                    current_chars += len(line) + 3
                else:
                    break
            content = header + "\n".join(f"- {line}" for line in truncated_lines)
            if len(truncated_lines) < len(lines):
                content += "\n- ... (more files)"
            token_estimate = self.config.max_artifact_tokens

        return ContextPart(
            content=content,
            source="artifact_surrogates",
            priority=8,  # High priority, after summary but before semantic
            token_estimate=token_estimate,
            metadata={
                "surrogate_count": len(surrogates),
            },
        )

    def assemble(
        self,
        summary: RunningSummary | None = None,
        recent_messages: list[ConversationMessage] | None = None,
        semantic_memories: list[MemoryFact] | None = None,
        artifact_surrogates: list[Any] | None = None,
        query: str | None = None,
    ) -> str:
        """Assemble complete context from all sources.

        This is the main method that builds context following the configured
        ordering and size limits.

        Args:
            summary: Optional running summary.
            recent_messages: Optional list of recent conversation messages.
            semantic_memories: Optional list of semantic memories from search.
            artifact_surrogates: Optional list of artifact text surrogates.
            query: Optional search query for context.

        Returns:
            Assembled context string ready for LLM prompt.
        """
        # Build all available parts
        parts: dict[str, ContextPart | None] = {
            "summary": self.build_summary_part(summary),
            "recent_messages": self.build_recent_messages_part(recent_messages or []),
            "semantic_memory": self.build_semantic_memory_part(semantic_memories or [], query),
            "artifact_surrogates": self.build_artifact_surrogates_part(artifact_surrogates or []),
        }

        # Order parts according to config
        ordered_parts: list[ContextPart] = []
        for source in self.config.order:
            part = parts.get(source)
            if part:
                ordered_parts.append(part)

        # Add any remaining parts not in order config
        for source, part in parts.items():
            if part and source not in self.config.order:
                ordered_parts.append(part)

        # Prune to total token limit
        pruned_parts = self._prune_to_token_limit(
            ordered_parts,
            self.config.max_total_tokens,
        )

        # Assemble final context
        sections = []
        for part in pruned_parts:
            sections.append(part.content)

        result = "\n\n".join(sections)

        logger.debug(
            "Assembled context: {} parts, {} estimated tokens",
            len(pruned_parts),
            sum(p.token_estimate for p in pruned_parts),
        )

        return result

    def assemble_for_llm(
        self,
        summary: RunningSummary | None = None,
        recent_messages: list[ConversationMessage] | None = None,
        semantic_memories: list[MemoryFact] | None = None,
        artifact_surrogates: list[Any] | None = None,
        query: str | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Assemble context as list of messages for LLM chat API.

        Args:
            summary: Optional running summary.
            recent_messages: Optional list of recent conversation messages.
            semantic_memories: Optional list of semantic memories from search.
            artifact_surrogates: Optional list of artifact text surrogates.
            query: Optional search query for context.
            system_prompt: Optional system prompt to prepend.

        Returns:
            List of message dictionaries for LLM API.
        """
        messages: list[dict[str, str]] = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Assemble context as a system message
        context = self.assemble(summary, recent_messages, semantic_memories, artifact_surrogates, query)
        if context:
            context_header = "Relevant context for this conversation:\n\n"
            messages.append(
                {
                    "role": "system",
                    "content": context_header + context,
                }
            )

        # Add recent messages as conversation history
        if recent_messages:
            for msg in recent_messages[-self.config.max_recent_messages :]:
                messages.append(msg.to_dict())

        return messages


# Global instance for dependency injection
_context_builder: ContextBuilder | None = None


def get_context_builder(config: ContextAssemblyConfig | None = None) -> ContextBuilder:
    """Get or create global ContextBuilder instance.

    Args:
        config: Optional configuration to use.

    Returns:
        ContextBuilder instance.
    """
    global _context_builder
    if _context_builder is None or config is not None:
        _context_builder = ContextBuilder(config)
    return _context_builder


def set_context_builder(builder: ContextBuilder | None) -> None:
    """Set global ContextBuilder instance (useful for testing).

    Args:
        builder: ContextBuilder instance or None to reset.
    """
    global _context_builder
    _context_builder = builder
