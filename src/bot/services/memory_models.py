"""Enhanced memory models for living bot experience."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional


class MemoryCategory(Enum):
    """High-level memory category for organization."""
    EPISODIC = "episodic"          # Specific events/conversations
    SEMANTIC = "semantic"          # General facts about user
    EMOTIONAL = "emotional"        # User moods, preferences, reactions
    PROCEDURAL = "procedural"      # How user likes things done
    PREFERENCE = "preference"      # User likes/dislikes
    RELATIONSHIP = "relationship"  # Bot-user relationship milestones


class MemoryType(Enum):
    """Specific memory type within category."""
    # Episodic
    CONVERSATION = "conversation"
    EVENT = "event"
    SHARED_EXPERIENCE = "shared_experience"
    
    # Semantic
    FACT = "fact"
    IDENTITY = "identity"          # Name, job, etc.
    GOAL = "goal"                  # User goals/aspirations
    
    # Emotional
    MOOD_STATE = "mood_state"
    EMOTIONAL_REACTION = "emotional_reaction"
    STRESS_EVENT = "stress_event"
    JOY_EVENT = "joy_event"
    
    # Preference
    LIKE = "like"
    DISLIKE = "dislike"
    TOPIC_INTEREST = "topic_interest"
    COMMUNICATION_STYLE = "communication_style"
    
    # Procedural
    HABIT = "habit"
    ROUTINE = "routine"
    
    # Relationship
    MILESTONE = "milestone"
    INSIDE_JOKE = "inside_joke"
    BOUNDARY = "boundary"
    
    # Media (existing)
    TEXT = "text"
    IMAGE = "image"
    GENERATED_IMAGE = "generated_image"


@dataclass
class EmotionalSnapshot:
    """Captures emotional state at a point in time."""
    valence: float = 0.0  # -1.0 to 1.0 (negative to positive)
    arousal: float = 0.0  # 0.0 to 1.0 (calm to excited)
    primary_emotion: str = "neutral"  # e.g., "joy", "sadness", "anger"
    confidence: float = 0.5  # 0.0 to 1.0


@dataclass
class MemoryFact:
    """Enhanced memory fact with rich metadata for living bot experience."""
    content: str
    user_id: int
    memory_type: MemoryType = MemoryType.TEXT
    memory_category: MemoryCategory = MemoryCategory.EPISODIC
    
    # Core metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    fact_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Importance & Decay
    importance_score: float = 1.0  # 0.0 to 2.0
    decay_rate: float = 0.01       # Daily decay rate
    expiration_date: Optional[datetime] = None
    
    # Emotional context
    emotional_valence: float = 0.0  # -1.0 (negative) to +1.0 (positive)
    emotional_arousal: float = 0.0  # 0.0 (calm) to 1.0 (excited)
    user_mood: Optional[str] = None  # e.g., "happy", "stressed", "tired"
    
    # Relationship context
    relationship_depth: int = 0     # 0 (stranger) to 10 (intimate friend)
    
    # Content specifics
    image_description: Optional[str] = None
    image_url: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    
    # Context links
    related_memories: list[str] = field(default_factory=list)  # fact_ids
    conversation_id: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expiration_date:
            return datetime.now() > self.expiration_date
        return False
    
    def get_effective_importance(self) -> float:
        """Calculate current importance considering decay."""
        days_since_creation = (datetime.now() - self.timestamp).days
        decayed = self.importance_score * (1 - self.decay_rate) ** max(days_since_creation, 0)
        # Boost for frequently accessed memories
        access_boost = min(self.access_count * 0.05, 0.3)
        return min(decayed + access_boost, 2.0)
    
    def to_mem0_metadata(self) -> dict[str, Any]:
        """Convert to Mem0-compatible metadata."""
        return {
            "memory_type": self.memory_type.value,
            "memory_category": self.memory_category.value,
            "importance_score": self.importance_score,
            "decay_rate": self.decay_rate,
            "emotional_valence": self.emotional_valence,
            "emotional_arousal": self.emotional_arousal,
            "user_mood": self.user_mood,
            "relationship_depth": self.relationship_depth,
            "tags": self.tags,
            "related_memories": self.related_memories,
            "conversation_id": self.conversation_id,
            "image_description": self.image_description,
            "image_url": self.image_url,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            **self.metadata
        }
    
    @classmethod
    def from_mem0_result(cls, result: dict[str, Any], user_id: int) -> "MemoryFact":
        """Create MemoryFact from Mem0 search result."""
        metadata = result.get("metadata", {})
        created_at = result.get("created_at")
        
        timestamp = datetime.now()
        if created_at:
            try:
                timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        
        # Parse last_accessed
        last_accessed = timestamp
        if "last_accessed" in metadata:
            try:
                last_accessed = datetime.fromisoformat(metadata["last_accessed"])
            except (ValueError, AttributeError):
                pass
        
        # Parse expiration_date
        expiration = None
        if "expiration_date" in metadata and metadata["expiration_date"]:
            try:
                expiration = datetime.fromisoformat(metadata["expiration_date"])
            except (ValueError, AttributeError):
                pass
        
        return cls(
            content=result.get("memory", ""),
            user_id=user_id,
            memory_type=MemoryType(metadata.get("memory_type", "text")),
            memory_category=MemoryCategory(metadata.get("memory_category", "episodic")),
            fact_id=result.get("id", ""),
            timestamp=timestamp,
            last_accessed=last_accessed,
            access_count=metadata.get("access_count", 0),
            importance_score=metadata.get("importance_score", 1.0),
            decay_rate=metadata.get("decay_rate", 0.01),
            expiration_date=expiration,
            emotional_valence=metadata.get("emotional_valence", 0.0),
            emotional_arousal=metadata.get("emotional_arousal", 0.0),
            user_mood=metadata.get("user_mood"),
            relationship_depth=metadata.get("relationship_depth", 0),
            tags=metadata.get("tags", []),
            related_memories=metadata.get("related_memories", []),
            conversation_id=metadata.get("conversation_id"),
            image_description=metadata.get("image_description"),
            image_url=metadata.get("image_url"),
            metadata={k: v for k, v in metadata.items() 
                     if k not in ["memory_type", "memory_category", "importance_score",
                                 "decay_rate", "emotional_valence", "emotional_arousal",
                                 "user_mood", "relationship_depth", "tags", "related_memories",
                                 "conversation_id", "image_description", "image_url",
                                 "last_accessed", "access_count", "expiration_date"]}
        )


@dataclass
class UserProfile:
    """Aggregated semantic knowledge about the user."""
    user_id: int
    
    # Identity
    name: Optional[str] = None
    preferred_name: Optional[str] = None
    age: Optional[int] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    
    # Preferences
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    communication_style: Optional[str] = None
    
    # Patterns
    common_topics: list[str] = field(default_factory=list)
    activity_patterns: dict[str, Any] = field(default_factory=dict)
    
    # Relationship
    relationship_level: int = 0  # 0-10
    shared_jokes: list[str] = field(default_factory=list)
    
    def to_context_string(self) -> str:
        """Convert profile to context string for LLM."""
        parts = []
        
        if self.preferred_name or self.name:
            parts.append(f"User's name: {self.preferred_name or self.name}")
        if self.occupation:
            parts.append(f"Occupation: {self.occupation}")
        if self.location:
            parts.append(f"Location: {self.location}")
        
        if self.interests:
            parts.append(f"Interests: {', '.join(self.interests[:5])}")
        
        if self.likes:
            parts.append(f"Likes: {', '.join(self.likes[:5])}")
        if self.dislikes:
            parts.append(f"Dislikes: {', '.join(self.dislikes[:3])}")
        
        if self.communication_style:
            parts.append(f"Communication style: {self.communication_style}")
        
        if self.relationship_level > 0:
            parts.append(f"Relationship level: {self.relationship_level}/10")
        
        return "\n".join(parts) if parts else ""


@dataclass
class ProactivePrompt:
    """A prompt for the bot to initiate conversation."""
    trigger_reason: str
    prompt_text: str
    priority: int  # 1-10
    context_memories: list[str]  # fact_ids that triggered this
