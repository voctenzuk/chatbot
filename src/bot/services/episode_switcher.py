"""Auto episode switching service for conversation management.

This module provides automatic episode switching based on:
- Time-gap trigger: Switch when too much time has passed between messages
- Topic shift detection: Detect topic changes using embedding similarity
- Anti-flap mechanism: Prevent rapid episode switching
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np
from loguru import logger


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into vectors."""
        ...


class SimpleEmbeddingProvider:
    """Simple TF-IDF based embedding provider (no external API needed).
    
    This is a fallback embedding provider that uses a simple bag-of-words
    approach with TF-IDF weighting. Good for testing and local development.
    """

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}
        self._doc_count: int = 0

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization (lowercase, split on whitespace/punctuation)."""
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def _get_tfidf_vector(self, text: str, vocab: dict[str, int], idf: dict[str, float]) -> np.ndarray:
        """Compute TF-IDF vector for text."""
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(len(vocab))

        # Term frequency
        tf: dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # Normalize TF
        total_tokens = len(tokens)
        for token in tf:
            tf[token] /= total_tokens

        # TF-IDF vector
        vector = np.zeros(len(vocab))
        for token, tf_val in tf.items():
            if token in vocab:
                vector[vocab[token]] = tf_val * idf.get(token, 1.0)

        return vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using TF-IDF.
        
        For a single text or first call, builds vocabulary from all texts.
        For consistency across calls in the same session, vocabulary accumulates.
        """
        if not texts:
            return []

        # Build vocabulary from all texts
        all_tokens: set[str] = set()
        tokenized_texts = []
        for text in texts:
            tokens = self._tokenize(text)
            tokenized_texts.append(tokens)
            all_tokens.update(tokens)

        # Update global vocabulary
        for token in all_tokens:
            if token not in self._vocabulary:
                self._vocabulary[token] = len(self._vocabulary)

        # Compute IDF
        self._doc_count += len(texts)
        idf: dict[str, float] = {}
        for token in all_tokens:
            doc_freq = sum(1 for tokens in tokenized_texts if token in tokens)
            idf[token] = np.log((self._doc_count + 1) / (doc_freq + 1)) + 1

        # Compute TF-IDF vectors
        vectors = []
        for text in texts:
            vector = self._get_tfidf_vector(text, self._vocabulary, idf)
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors.append(vector.tolist())

        return vectors


class OpenAIEmbeddingProvider:
    """OpenAI/OpenRouter embedding provider."""

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.model = model
        self._client = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                )
            except ImportError:
                raise RuntimeError("openai package is required for OpenAIEmbeddingProvider")

        response = await self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]


@dataclass
class EpisodeConfig:
    """Configuration for episode switching behavior."""

    # Time gap trigger (seconds)
    time_gap_threshold: float = 3600.0  # 1 hour default

    # Topic shift detection
    topic_similarity_threshold: float = 0.7  # Cosine similarity threshold
    topic_min_messages: int = 3  # Minimum messages before topic shift can trigger

    # Anti-flap protection
    min_episode_duration: float = 300.0  # 5 minutes minimum episode duration
    max_episodes_per_hour: int = 10  # Rate limiting

    # Combined scoring weights
    time_gap_weight: float = 0.4
    topic_shift_weight: float = 0.6


@dataclass
class Message:
    """A message in the conversation."""

    content: str
    timestamp: datetime
    user_id: int
    role: str = "user"  # "user" or "assistant"
    embedding: list[float] | None = None


@dataclass
class Episode:
    """A conversation episode (a contiguous segment of conversation)."""

    episode_id: str
    user_id: int
    start_time: datetime
    end_time: datetime
    messages: list[Message] = field(default_factory=list)
    summary: str | None = None
    topic_keywords: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Get episode duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def message_count(self) -> int:
        """Get number of messages in episode."""
        return len(self.messages)

    def get_combined_text(self) -> str:
        """Get combined text of all messages for embedding comparison."""
        return " ".join(m.content for m in self.messages)


@dataclass
class SwitchDecision:
    """Decision result from episode switch evaluation."""

    should_switch: bool
    reason: str
    confidence: float  # 0.0 to 1.0
    trigger_type: str | None = None  # "time_gap", "topic_shift", "combined"


class EpisodeManager:
    """Manages conversation episodes with auto-switching logic.
    
    Features:
    - Time-gap trigger: Switch when too much time passes between messages
    - Topic shift detection: Use embedding similarity to detect topic changes
    - Anti-flap protection: Prevent rapid episode switching
    """

    def __init__(
        self,
        config: EpisodeConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config or EpisodeConfig()
        self.embedding_provider = embedding_provider or SimpleEmbeddingProvider()
        self._episodes: dict[int, list[Episode]] = {}
        self._current_episode: dict[int, Episode | None] = {}
        self._episode_counter: int = 0
        self._switch_history: dict[int, list[datetime]] = {}  # For anti-flap

    def _generate_episode_id(self) -> str:
        """Generate a unique episode ID."""
        self._episode_counter += 1
        return f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._episode_counter}"

    def _get_recent_switch_count(self, user_id: int, window_hours: float = 1.0) -> int:
        """Count recent episode switches for anti-flap."""
        if user_id not in self._switch_history:
            return 0

        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent = [t for t in self._switch_history[user_id] if t > cutoff]
        self._switch_history[user_id] = recent  # Clean up old entries
        return len(recent)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        vec_a = np.array(a)
        vec_b = np.array(b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    async def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding for text."""
        embeddings = await self.embedding_provider.embed([text])
        return embeddings[0]

    def _check_time_gap(self, last_message_time: datetime, current_time: datetime) -> tuple[bool, float]:
        """Check if time gap exceeds threshold.
        
        Returns:
            (should_trigger, confidence_score)
        """
        gap_seconds = (current_time - last_message_time).total_seconds()

        if gap_seconds >= self.config.time_gap_threshold:
            # Normalize confidence: 1.0 at threshold, approaches 1.0 asymptotically
            confidence = min(1.0, 0.5 + (gap_seconds / self.config.time_gap_threshold) * 0.5)
            return True, confidence

        # Partial confidence based on how close to threshold
        confidence = gap_seconds / self.config.time_gap_threshold * 0.5
        return False, confidence

    async def _check_topic_shift(
        self,
        current_episode: Episode,
        new_message: Message,
    ) -> tuple[bool, float]:
        """Check if new message represents a topic shift.
        
        Returns:
            (should_trigger, confidence_score)
        """
        # Need minimum messages before topic shift can trigger
        if current_episode.message_count < self.config.topic_min_messages:
            return False, 0.0

        # Get embeddings
        if not current_episode.messages:
            return False, 0.0

        # Use last few messages from current episode as topic reference
        recent_messages = current_episode.messages[-5:]  # Last 5 messages
        episode_text = " ".join(m.content for m in recent_messages)

        # Compute embeddings
        embeddings = await self.embedding_provider.embed([episode_text, new_message.content])
        episode_embedding = embeddings[0]
        message_embedding = embeddings[1]

        # Store embedding in message for future reference
        new_message.embedding = message_embedding

        similarity = self._cosine_similarity(episode_embedding, message_embedding)

        # Similarity below threshold indicates topic shift
        if similarity < self.config.topic_similarity_threshold:
            # Confidence increases as similarity decreases
            confidence = min(1.0, (self.config.topic_similarity_threshold - similarity) / 0.5)
            return True, confidence

        return False, 0.0

    async def evaluate_switch(
        self,
        user_id: int,
        message_content: str,
        timestamp: datetime | None = None,
    ) -> SwitchDecision:
        """Evaluate whether to start a new episode.
        
        Args:
            user_id: User identifier
            message_content: Content of the new message
            timestamp: Message timestamp (defaults to now)

        Returns:
            SwitchDecision with should_switch and reasoning
        """
        current_time = timestamp or datetime.now()

        # Anti-flap: Check rate limiting
        recent_switches = self._get_recent_switch_count(user_id)
        if recent_switches >= self.config.max_episodes_per_hour:
            return SwitchDecision(
                should_switch=False,
                reason=f"Rate limit: {recent_switches} switches in the last hour",
                confidence=1.0,
                trigger_type="anti_flap_rate_limit",
            )

        current_episode = self._current_episode.get(user_id)

        # No current episode - start one
        if current_episode is None or not current_episode.messages:
            return SwitchDecision(
                should_switch=True,
                reason="Starting first episode",
                confidence=1.0,
                trigger_type="new_conversation",
            )

        # Anti-flap: Minimum episode duration
        time_since_start = (current_time - current_episode.start_time).total_seconds()
        if time_since_start < self.config.min_episode_duration:
            return SwitchDecision(
                should_switch=False,
                reason=f"Episode too young ({time_since_start:.0f}s < {self.config.min_episode_duration}s)",
                confidence=1.0,
                trigger_type="anti_flap_min_duration",
            )

        last_message = current_episode.messages[-1]

        # Check time gap
        time_trigger, time_confidence = self._check_time_gap(last_message.timestamp, current_time)

        # Check topic shift
        new_message = Message(
            content=message_content,
            timestamp=current_time,
            user_id=user_id,
        )
        topic_trigger, topic_confidence = await self._check_topic_shift(current_episode, new_message)

        # Combined decision
        combined_score = (
            time_confidence * self.config.time_gap_weight +
            topic_confidence * self.config.topic_shift_weight
        )

        # Decision logic
        if time_trigger and topic_trigger:
            return SwitchDecision(
                should_switch=True,
                reason=f"Time gap ({(current_time - last_message.timestamp).total_seconds()/60:.0f}min) and topic shift detected",
                confidence=combined_score,
                trigger_type="combined",
            )
        elif time_trigger and time_confidence > 0.7:
            return SwitchDecision(
                should_switch=True,
                reason=f"Significant time gap: {(current_time - last_message.timestamp).total_seconds()/60:.0f} minutes",
                confidence=time_confidence,
                trigger_type="time_gap",
            )
        elif topic_trigger and topic_confidence > 0.6:
            return SwitchDecision(
                should_switch=True,
                reason="Significant topic shift detected",
                confidence=topic_confidence,
                trigger_type="topic_shift",
            )

        return SwitchDecision(
            should_switch=False,
            reason="Continuing current episode",
            confidence=1.0 - combined_score,
            trigger_type=None,
        )

    async def add_message(
        self,
        user_id: int,
        content: str,
        role: str = "user",
        timestamp: datetime | None = None,
    ) -> tuple[Message, SwitchDecision]:
        """Add a message and handle episode switching.
        
        Args:
            user_id: User identifier
            content: Message content
            role: "user" or "assistant"
            timestamp: Message timestamp

        Returns:
            Tuple of (Message, SwitchDecision)
        """
        current_time = timestamp or datetime.now()

        # Evaluate switch
        decision = await self.evaluate_switch(user_id, content, current_time)

        # Start new episode if needed
        if decision.should_switch:
            await self._start_new_episode(user_id, current_time)

        # Create and add message
        message = Message(
            content=content,
            timestamp=current_time,
            user_id=user_id,
            role=role,
        )

        current_episode = self._current_episode.get(user_id)
        if current_episode:
            current_episode.messages.append(message)
            current_episode.end_time = current_time

            # Compute embedding for topic tracking
            if len(current_episode.messages) == 1:
                message.embedding = await self._compute_embedding(content)

        return message, decision

    async def _start_new_episode(self, user_id: int, start_time: datetime) -> Episode:
        """Start a new episode for a user."""
        # Record switch for anti-flap
        if user_id not in self._switch_history:
            self._switch_history[user_id] = []
        self._switch_history[user_id].append(start_time)

        # Create new episode
        episode = Episode(
            episode_id=self._generate_episode_id(),
            user_id=user_id,
            start_time=start_time,
            end_time=start_time,
            messages=[],
        )

        # Store reference
        if user_id not in self._episodes:
            self._episodes[user_id] = []

        # Archive current episode if exists
        current = self._current_episode.get(user_id)
        if current and current.messages:
            self._episodes[user_id].append(current)
            logger.debug(
                "Archived episode {} for user {} ({} messages, {}s duration)",
                current.episode_id,
                user_id,
                current.message_count,
                current.duration_seconds,
            )

        self._current_episode[user_id] = episode

        logger.info(
            "Started new episode {} for user {}",
            episode.episode_id,
            user_id,
        )

        return episode

    def get_current_episode(self, user_id: int) -> Episode | None:
        """Get the current episode for a user."""
        return self._current_episode.get(user_id)

    def get_episodes(self, user_id: int) -> list[Episode]:
        """Get all archived episodes for a user."""
        return self._episodes.get(user_id, []).copy()

    def get_all_episodes(self, user_id: int) -> list[Episode]:
        """Get all episodes including current for a user."""
        episodes = self.get_episodes(user_id)
        current = self.get_current_episode(user_id)
        if current:
            return episodes + [current]
        return episodes

    def clear_user_episodes(self, user_id: int) -> None:
        """Clear all episodes for a user."""
        self._episodes.pop(user_id, None)
        self._current_episode.pop(user_id, None)
        self._switch_history.pop(user_id, None)
        logger.debug("Cleared all episodes for user {}", user_id)

    async def get_episode_summary(self, episode: Episode) -> str:
        """Generate a summary for an episode.
        
        For now, returns a simple text summary. Could be enhanced with LLM.
        """
        if not episode.messages:
            return "Empty episode"

        topics = []
        if episode.topic_keywords:
            topics = episode.topic_keywords
        else:
            # Simple keyword extraction (first few significant words)
            all_text = episode.get_combined_text().lower()
            # Filter common stop words (English and Russian)
            stop_words = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                'through', 'during', 'before', 'after', 'above', 'below',
                'between', 'under', 'and', 'but', 'or', 'yet', 'so',
                'if', 'because', 'although', 'though', 'while', 'where',
                'when', 'that', 'which', 'who', 'whom', 'whose', 'what',
                'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
                'we', 'they', 'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'its', 'our', 'their',
                'и', 'в', 'не', 'на', 'я', 'быть', 'он', 'с', 'что',
                'а', 'по', 'это', 'она', 'к', 'но', 'мы', 'как',
                'из', 'у', 'то', 'за', 'свой', 'ее', 'который',
                'весь', 'год', 'от', 'так', 'о', 'для', 'ты', 'же',
                'все', 'тот', 'мочь', 'вы', 'человек', 'такой',
            }
            words = [w for w in all_text.split() if len(w) > 3 and w not in stop_words]
            from collections import Counter
            topics = [word for word, _ in Counter(words).most_common(5)]

        duration_mins = episode.duration_seconds / 60
        return (
            f"Episode {episode.episode_id}: {episode.message_count} messages, "
            f"{duration_mins:.1f}min duration"
            f"{f' (topics: {', '.join(topics)})' if topics else ''}"
        )


# Global instance for dependency injection
_episode_manager: EpisodeManager | None = None


def get_episode_manager() -> EpisodeManager:
    """Get or create global episode manager instance."""
    global _episode_manager
    if _episode_manager is None:
        _episode_manager = EpisodeManager()
    return _episode_manager


def set_episode_manager(manager: EpisodeManager | None) -> None:
    """Set global episode manager instance (useful for testing)."""
    global _episode_manager
    _episode_manager = manager
