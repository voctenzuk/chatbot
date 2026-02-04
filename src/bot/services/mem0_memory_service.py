"""Mem0 Memory Service wrapper for OSS + pgvector integration.

This module provides the Mem0MemoryService class that interfaces with the Mem0 OSS API
for long-term memory storage and retrieval. It supports:
- Factual memory storage (semantic/user facts)
- Episodic memory storage (conversation history with run_id)
- Vector search using pgvector or other backends

Configuration (via environment variables or .env file):
    MEM0_API_KEY: Your Mem0 API key (required for cloud, optional for OSS)
    MEM0_HOST: Mem0 API host URL (e.g., http://localhost:8000 for OSS)
    MEM0_PROJECT_ID: Mem0 project ID (optional)
    MEM0_ORG_ID: Mem0 organization ID (optional)

Vector Store Configuration (pgvector):
    See config notes below for pgvector setup.

Ingestion Guidelines:
    See ingestion instructions below for what to store/ignore.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol
from unittest.mock import MagicMock

from loguru import logger

from bot.config import settings
from bot.services.memory_models import MemoryCategory, MemoryFact, MemoryType

# Import mem0 client - may not be available in all environments
try:
    from mem0 import MemoryClient

    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    MemoryClient = None  # type: ignore[misc, assignment]


class Mem0ClientProtocol(Protocol):
    """Protocol for Mem0 client to enable mocking/testing."""

    def add(
        self,
        messages: str | dict[str, Any] | list[dict[str, Any]],
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a memory."""
        ...

    def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search memories."""
        ...

    def get_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get all memories."""
        ...

    def delete_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Delete all memories."""
        ...


"""
================================================================================
PGVECTOR CONFIGURATION NOTES
================================================================================

To use pgvector as the vector store backend for Mem0 OSS:

1. Install pgvector extension in PostgreSQL:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. Configure Mem0 OSS with pgvector in your Mem0 config (config.yaml or env vars):
   ```yaml
   vector_store:
     provider: pgvector
     config:
       host: localhost
       port: 5432
       dbname: mem0
       user: postgres
       password: your_password
       embedding_model_dims: 1536  # Match your embedding model dimensions
   ```

3. Environment variables for pgvector:
   - MEM0_VECTOR_STORE_PROVIDER=pgvector
   - MEM0_VECTOR_STORE_HOST=localhost
   - MEM0_VECTOR_STORE_PORT=5432
   - MEM0_VECTOR_STORE_DBNAME=mem0
   - MEM0_VECTOR_STORE_USER=postgres
   - MEM0_VECTOR_STORE_PASSWORD=your_password
   - MEM0_VECTOR_STORE_EMBEDDING_MODEL_DIMS=1536

4. Required PostgreSQL schema (Mem0 OSS will create this automatically,
   but you can verify):
   ```sql
   CREATE TABLE IF NOT EXISTS memories (
       id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
       user_id TEXT,
       agent_id TEXT,
       run_id TEXT,
       content TEXT,
       embedding vector(1536),
       metadata JSONB,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops);
   CREATE INDEX ON memories (user_id);
   CREATE INDEX ON memories (run_id);
   ```

5. Docker Compose example:
   ```yaml
   services:
     postgres:
       image: ankane/pgvector:latest
       environment:
         POSTGRES_DB: mem0
         POSTGRES_USER: postgres
         POSTGRES_PASSWORD: your_password
       ports:
         - "5432:5432"
       volumes:
         - postgres_data:/var/lib/postgresql/data

     mem0:
       image: mem0/mem0:latest
       environment:
         - MEM0_VECTOR_STORE_PROVIDER=pgvector
         - MEM0_VECTOR_STORE_HOST=postgres
         - MEM0_VECTOR_STORE_PORT=5432
         - MEM0_VECTOR_STORE_DBNAME=mem0
         - MEM0_VECTOR_STORE_USER=postgres
         - MEM0_VECTOR_STORE_PASSWORD=your_password
       ports:
         - "8000:8000"
       depends_on:
         - postgres
   ```

================================================================================
"""

"""
================================================================================
INGESTION INSTRUCTIONS - WHAT TO STORE/IGNORE
================================================================================

WHAT TO STORE (High Value Memories):
------------------------------------

1. USER FACTS (Semantic Memories - write_factual):
   - Personal information: name, age, occupation, location
   - Preferences: likes, dislikes, interests, hobbies
   - Goals: short-term and long-term aspirations
   - Relationships: family, friends, pets
   - Important dates: birthdays, anniversaries
   
   Example: "User's name is Alex, works as a software engineer, likes hiking"

2. CONVERSATION EVENTS (Episodic Memories - write_episodic):
   - Significant shared experiences
   - Important decisions or conclusions
   - Emotional moments (joy, frustration, excitement)
   - Questions answered with useful information
   - Inside jokes or shared references
   
   Example: "User and I discussed Python async patterns, user was excited about asyncio"

3. CONTEXTUAL INFORMATION:
   - Current projects user is working on
   - Recent life events (new job, moved, etc.)
   - Preferences discovered during conversation

WHAT TO IGNORE (Low Value / Noise):
------------------------------------

1. TRANSIENT CONTENT:
   - Greetings: "Hi", "Hello", "Good morning"
   - Acknowledgments: "OK", "Thanks", "I see"
   - Small talk without substance
   
2. REDUNDANT INFORMATION:
   - Facts already stored (check before storing)
   - Repetitive questions/answers
   - System messages or bot responses without user context

3. SENSITIVE DATA (Never Store):
   - Passwords or credentials
   - Financial details (credit cards, bank accounts)
   - Private information user explicitly wants forgotten
   - Personal identifiable information of third parties

4. TEMPORAL NOISE:
   - Current date/time references ("today", "yesterday")
   - Weather unless user has specific preferences
   - News headlines (unless user shows sustained interest)

INGESTION BEST PRACTICES:
-------------------------

1. Summarize: Store condensed versions of long conversations
2. Extract entities: Pull out key names, places, concepts
3. Tag appropriately: Use metadata tags for categorization
4. Set importance: Mark critical memories with higher importance
5. Set expiration: Temporary info can have expiration dates
6. Deduplicate: Check for existing similar memories before storing

MEMORY PRIORITY RANKING:
------------------------

High Priority (Always store):
- User identity information
- Explicit preferences stated by user
- Important life events
- Emotional milestones

Medium Priority (Store selectively):
- Conversation summaries
- Topic interests
- Casual mentions of preferences

Low Priority (Usually ignore):
- One-off questions
- Greetings and farewells
- Acknowledgment messages

================================================================================
"""


class Mem0MemoryService:
    """Memory service using Mem0 OSS/Cloud for long-term storage with vector search.

    This service provides:
    - Factual memory storage (semantic/user facts) via write_factual()
    - Episodic memory storage (conversation events) via write_episodic()
    - Vector-based semantic search via search()
    - Metadata filtering and categorization
    - Support for pgvector and other vector backends

    Methods:
        write_factual: Store factual/semantic memories about the user
        write_episodic: Store episodic memories tied to a conversation run
        search: Search memories using semantic vector search
    """

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        org_id: str | None = None,
        host: str | None = None,
        client: Mem0ClientProtocol | None = None,
    ) -> None:
        """Initialize Mem0 Memory client with configuration.

        Args:
            api_key: Mem0 API key. If not provided, reads from settings or env var.
            project_id: Mem0 project ID. Optional.
            org_id: Mem0 organization ID. Optional.
            host: Mem0 API host URL. For OSS: http://localhost:8000
            client: Optional pre-configured client (for testing/mocking).

        Raises:
            RuntimeError: If mem0 package is not installed or API key is missing
                         (for cloud deployments).
        """
        self._client: Mem0ClientProtocol
        self._is_mock = False

        if client is not None:
            # Use provided client (for testing)
            self._client = client
            self._is_mock = isinstance(client, MagicMock)
            logger.info("Mem0MemoryService initialized with custom client")
            return

        if not MEM0_AVAILABLE:
            raise RuntimeError(
                "mem0 package is not installed. "
                "Install it with: pip install mem0ai"
            )

        # Get configuration from parameters or settings/env vars
        self._api_key = api_key or settings.mem0_api_key or os.getenv("MEM0_API_KEY")
        self._project_id = project_id or settings.mem0_project_id or os.getenv("MEM0_PROJECT_ID")
        self._org_id = org_id or os.getenv("MEM0_ORG_ID")
        self._host = host or os.getenv("MEM0_HOST")

        # API key is optional for OSS deployments (local Mem0)
        # but required for cloud
        if not self._api_key and not self._host:
            raise RuntimeError(
                "Mem0 API key is required for cloud deployments. "
                "Set MEM0_API_KEY environment variable or provide host for OSS."
            )

        # Initialize the Mem0 client
        client_kwargs: dict[str, Any] = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        if self._project_id:
            client_kwargs["project_id"] = self._project_id
        if self._org_id:
            client_kwargs["org_id"] = self._org_id
        if self._host:
            client_kwargs["host"] = self._host

        self._client = MemoryClient(**client_kwargs)  # type: ignore[assignment]
        logger.info(
            "Mem0MemoryService initialized (host={}, project_id={})",
            self._host or "cloud",
            self._project_id or "default",
        )

    def _format_user_id(self, user_id: int) -> str:
        """Format user_id as string for Mem0 API.

        Args:
            user_id: Numeric user identifier.

        Returns:
            String user identifier with prefix for namespacing.
        """
        return f"tg_user_{user_id}"

    async def write_factual(
        self,
        content: str,
        user_id: int,
        metadata: dict[str, Any] | None = None,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 1.0,
        tags: list[str] | None = None,
    ) -> str:
        """Write a factual/semantic memory about the user.

        Use this for storing persistent facts about the user such as:
        - Personal information (name, age, occupation)
        - Preferences and interests
        - Goals and aspirations
        - Relationship information

        These memories are NOT tied to a specific conversation run.

        Args:
            content: The factual content to store.
            user_id: User identifier.
            metadata: Optional metadata about the fact.
            memory_type: Type of memory (default: FACT).
            importance: Importance score (0.0 to 2.0, default: 1.0).
            tags: Optional tags for categorization.

        Returns:
            Memory ID (fact_id from Mem0).
        """
        user_id_str = self._format_user_id(user_id)

        # Build metadata for factual memory
        fact_metadata = (metadata or {}).copy()
        fact_metadata.update({
            "memory_type": memory_type.value,
            "memory_category": MemoryCategory.SEMANTIC.value,
            "importance_score": importance,
            "tags": tags or [],
            "is_factual": True,
            "source": "telegram_bot",
        })

        try:
            # Use a user message format for the memory
            messages = {"role": "user", "content": content}

            response = self._client.add(
                messages=messages,
                user_id=user_id_str,
                metadata=fact_metadata,
            )

            # Extract memory ID from response
            results = response.get("results", [])
            if results and len(results) > 0:
                memory_id = results[0].get("id", "")
            else:
                memory_id = ""

            logger.debug(
                "Written factual memory {} for user {}: {}",
                memory_id,
                user_id,
                content[:50],
            )
            return memory_id

        except Exception as e:
            logger.error("Failed to write factual memory for user {}: {}", user_id, e)
            raise

    async def write_episodic(
        self,
        content: str,
        user_id: int,
        run_id: str,
        metadata: dict[str, Any] | None = None,
        memory_type: MemoryType = MemoryType.CONVERSATION,
        importance: float = 1.0,
        tags: list[str] | None = None,
        emotional_valence: float = 0.0,
    ) -> str:
        """Write an episodic memory tied to a specific conversation run.

        Use this for storing conversation events such as:
        - Significant exchanges in a conversation
        - Shared experiences during a chat session
        - Important decisions or conclusions
        - Emotional moments

        These memories ARE tied to a specific conversation run_id.

        Args:
            content: The episodic content to store.
            user_id: User identifier.
            run_id: Conversation run identifier (session ID).
            metadata: Optional metadata about the memory.
            memory_type: Type of memory (default: CONVERSATION).
            importance: Importance score (0.0 to 2.0, default: 1.0).
            tags: Optional tags for categorization.
            emotional_valence: Emotional tone (-1.0 negative to +1.0 positive).

        Returns:
            Memory ID (fact_id from Mem0).
        """
        user_id_str = self._format_user_id(user_id)

        # Build metadata for episodic memory
        episodic_metadata = (metadata or {}).copy()
        episodic_metadata.update({
            "memory_type": memory_type.value,
            "memory_category": MemoryCategory.EPISODIC.value,
            "importance_score": importance,
            "tags": tags or [],
            "is_episodic": True,
            "emotional_valence": emotional_valence,
            "source": "telegram_bot",
        })

        try:
            # Use a user message format for the memory
            messages = {"role": "user", "content": content}

            response = self._client.add(
                messages=messages,
                user_id=user_id_str,
                run_id=run_id,
                metadata=episodic_metadata,
            )

            # Extract memory ID from response
            results = response.get("results", [])
            if results and len(results) > 0:
                memory_id = results[0].get("id", "")
            else:
                memory_id = ""

            logger.debug(
                "Written episodic memory {} for user {} run {}: {}",
                memory_id,
                user_id,
                run_id,
                content[:50],
            )
            return memory_id

        except Exception as e:
            logger.error(
                "Failed to write episodic memory for user {} run {}: {}",
                user_id,
                run_id,
                e,
            )
            raise

    async def search(
        self,
        query: str,
        user_id: int,
        run_id: str | None = None,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryFact]:
        """Search memories using semantic vector search.

        Searches both factual and episodic memories for the user.
        Optionally filter by run_id to search within a specific conversation.

        Args:
            query: Search query (semantic search using vector similarity).
            user_id: User identifier.
            run_id: Optional run_id to filter by conversation.
            limit: Maximum number of results (default: 5).
            filters: Optional additional filters for metadata.

        Returns:
            List of matching MemoryFact objects.
        """
        user_id_str = self._format_user_id(user_id)

        try:
            search_kwargs: dict[str, Any] = {
                "query": query,
                "user_id": user_id_str,
                "limit": limit,
            }
            if run_id:
                search_kwargs["run_id"] = run_id
            if filters:
                search_kwargs["filters"] = filters

            response = self._client.search(**search_kwargs)

            results = response.get("results", [])
            memories: list[MemoryFact] = []

            for result in results:
                # Parse metadata
                metadata = result.get("metadata", {})
                memory_type_value = metadata.get("memory_type", "text")
                try:
                    memory_type = MemoryType(memory_type_value)
                except ValueError:
                    memory_type = MemoryType.TEXT

                memory_category_value = metadata.get("memory_category", "episodic")
                try:
                    memory_category = MemoryCategory(memory_category_value)
                except ValueError:
                    memory_category = MemoryCategory.EPISODIC

                # Parse timestamp
                created_at = result.get("created_at")
                if created_at:
                    try:
                        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()

                # Parse last_accessed
                last_accessed = timestamp
                if "last_accessed" in metadata:
                    try:
                        last_accessed = datetime.fromisoformat(metadata["last_accessed"])
                    except (ValueError, AttributeError):
                        pass

                memory = MemoryFact(
                    content=result.get("memory", ""),
                    user_id=user_id,
                    memory_type=memory_type,
                    memory_category=memory_category,
                    fact_id=result.get("id", ""),
                    timestamp=timestamp,
                    last_accessed=last_accessed,
                    access_count=metadata.get("access_count", 0),
                    importance_score=metadata.get("importance_score", 1.0),
                    emotional_valence=metadata.get("emotional_valence", 0.0),
                    tags=metadata.get("tags", []),
                    metadata=metadata,
                )
                memories.append(memory)

            logger.debug(
                "Search for '{}' returned {} memories for user {} (run_id={})",
                query,
                len(memories),
                user_id,
                run_id,
            )
            return memories

        except Exception as e:
            logger.error("Failed to search memories for user {}: {}", user_id, e)
            return []

    async def get_all_memories(
        self,
        user_id: int,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryFact]:
        """Get all memories for a user.

        Args:
            user_id: User identifier.
            run_id: Optional run_id to filter by conversation.
            limit: Maximum number of results (default: 100).

        Returns:
            List of MemoryFact objects.
        """
        user_id_str = self._format_user_id(user_id)

        try:
            get_kwargs: dict[str, Any] = {
                "user_id": user_id_str,
                "limit": limit,
            }
            if run_id:
                get_kwargs["run_id"] = run_id

            response = self._client.get_all(**get_kwargs)

            results = response.get("results", [])
            memories: list[MemoryFact] = []

            for result in results:
                metadata = result.get("metadata", {})
                memory_type_value = metadata.get("memory_type", "text")
                try:
                    memory_type = MemoryType(memory_type_value)
                except ValueError:
                    memory_type = MemoryType.TEXT

                memory_category_value = metadata.get("memory_category", "episodic")
                try:
                    memory_category = MemoryCategory(memory_category_value)
                except ValueError:
                    memory_category = MemoryCategory.EPISODIC

                created_at = result.get("created_at")
                if created_at:
                    try:
                        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()

                memory = MemoryFact(
                    content=result.get("memory", ""),
                    user_id=user_id,
                    memory_type=memory_type,
                    memory_category=memory_category,
                    fact_id=result.get("id", ""),
                    timestamp=timestamp,
                    metadata=metadata,
                )
                memories.append(memory)

            return memories

        except Exception as e:
            logger.error("Failed to get all memories for user {}: {}", user_id, e)
            return []

    async def delete_user_memories(
        self,
        user_id: int,
        run_id: str | None = None,
    ) -> None:
        """Delete all memories for a user.

        Args:
            user_id: User identifier.
            run_id: Optional run_id to delete only memories from a specific run.
        """
        user_id_str = self._format_user_id(user_id)

        try:
            delete_kwargs: dict[str, Any] = {"user_id": user_id_str}
            if run_id:
                delete_kwargs["run_id"] = run_id

            self._client.delete_all(**delete_kwargs)
            logger.info(
                "Deleted all memories for user {} (run_id={})",
                user_id,
                run_id,
            )
        except Exception as e:
            logger.error("Failed to delete memories for user {}: {}", user_id, e)
            raise


# Global instance for dependency injection
_memory_service: Mem0MemoryService | None = None


def get_memory_service() -> Mem0MemoryService:
    """Get or create global memory service instance.

    Returns:
        Mem0MemoryService instance.

    Raises:
        RuntimeError: If service cannot be initialized.
    """
    global _memory_service
    if _memory_service is None:
        _memory_service = Mem0MemoryService()
    return _memory_service


def set_memory_service(service: Mem0MemoryService | None) -> None:
    """Set global memory service instance (useful for testing).

    Args:
        service: Mem0MemoryService instance or None to reset.
    """
    global _memory_service
    _memory_service = service
