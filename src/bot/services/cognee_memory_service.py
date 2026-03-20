"""Cognee Memory Service wrapper for knowledge graph-based memory.

Replaces the previous Mem0MemoryService with Cognee's knowledge graph engine.
Cognee builds a structured knowledge graph from unstructured data, enabling
richer semantic search and relationship discovery.

Pipeline:
    1. cognee.add() — stores raw data as chunks (fast)
    2. cognee.cognify() — builds knowledge graph (heavy, run periodically)
    3. cognee.search() — searches via vector similarity (CHUNKS) or graph (GRAPH_COMPLETION)

Configuration (via environment variables or .env file):
    LLM_API_KEY: API key for the LLM provider (used by cognee for graph construction)
    LLM_BASE_URL: Custom LLM endpoint URL (optional)
    LLM_MODEL: LLM model name (optional)
    VECTOR_DB_PROVIDER: Vector database backend (default: lancedb)
    GRAPH_DATABASE_PROVIDER: Graph database backend (default: kuzu)
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Any, Protocol

from loguru import logger

from bot.config import settings
from bot.services.memory_models import MemoryCategory, MemoryFact, MemoryType

# Import cognee - may not be available in all environments
try:
    import cognee as _cognee_mod

    COGNEE_AVAILABLE = True
except ImportError:
    COGNEE_AVAILABLE = False
    _cognee_mod = None  # type: ignore[assignment]


class CogneeClientProtocol(Protocol):
    """Protocol for cognee client operations to enable testing."""

    async def add(self, data: str | list[str], dataset_name: str | None = None) -> None: ...
    async def cognify(self, datasets: list[str] | None = None) -> None: ...
    async def search(self, query_text: str, datasets: list[str] | None = None) -> list[Any]: ...
    async def delete_dataset(self, dataset_name: str) -> None: ...


class CogneeClient:
    """Thin wrapper around cognee module for testability."""

    def __init__(self) -> None:
        if _cognee_mod is None:
            raise RuntimeError("cognee package is not installed")
        self._cognee = _cognee_mod

    async def add(self, data: str | list[str], dataset_name: str | None = None) -> None:
        if dataset_name:
            await self._cognee.add(data, dataset_name)
        else:
            await self._cognee.add(data)

    async def cognify(self, datasets: list[str] | None = None) -> None:
        await self._cognee.cognify(datasets=datasets)

    async def search(self, query_text: str, datasets: list[str] | None = None) -> list[Any]:
        return await self._cognee.search(  # type: ignore[no-any-return]
            query_text, datasets=datasets
        )

    async def delete_dataset(self, dataset_name: str) -> None:
        await self._cognee.prune.prune_data(metadata=dataset_name)


class CogneeMemoryService:
    """Memory service using Cognee for knowledge graph-based storage and retrieval.

    This service provides the same interface as the previous Mem0MemoryService:
    - write_factual: Store factual/semantic memories about the user
    - write_episodic: Store episodic memories tied to a conversation run
    - search: Search memories using vector-based search
    - get_all_memories: Retrieve all memories for a user
    - delete_user_memories: Remove all memories for a user
    - cognify: Build/update the knowledge graph from stored data

    Data isolation is achieved via per-user datasets (tg_user_{user_id}).

    The cognify() step is NOT called automatically on writes for performance.
    Call cognify() periodically or after a batch of writes to build the
    knowledge graph for richer GRAPH_COMPLETION searches.
    """

    def __init__(
        self,
        client: CogneeClientProtocol | None = None,
    ) -> None:
        """Initialize Cognee Memory client.

        Args:
            client: Optional pre-configured client (for testing/mocking).

        Raises:
            RuntimeError: If cognee package is not installed.
        """
        self._client: CogneeClientProtocol
        self._pending_datasets: set[str] = set()
        self._cognify_lock = asyncio.Lock()

        if client is not None:
            self._client = client
            logger.info("CogneeMemoryService initialized with custom client")
            return

        if not COGNEE_AVAILABLE:
            raise RuntimeError(
                "cognee package is not installed. Install it with: pip install cognee"
            )

        self._configure()
        self._client = CogneeClient()
        logger.info("CogneeMemoryService initialized")

    def _configure(self) -> None:
        """Configure cognee from application settings."""
        if _cognee_mod is None:
            return

        api_key = settings.llm_api_key
        if api_key:
            _cognee_mod.config.set_llm_api_key(api_key)

        if settings.llm_model:
            _cognee_mod.config.set_llm_model(settings.llm_model)

        if settings.llm_base_url:
            _cognee_mod.config.set_llm_endpoint(settings.llm_base_url)

    def _user_dataset(self, user_id: int) -> str:
        """Format dataset name for user isolation.

        Args:
            user_id: Numeric user identifier.

        Returns:
            Dataset name string.
        """
        return f"tg_user_{user_id}"

    @staticmethod
    def _generate_memory_id(content: str, user_id: int) -> str:
        """Generate a deterministic-ish memory ID.

        Args:
            content: Memory content.
            user_id: User identifier.

        Returns:
            Short hex hash string.
        """
        hash_input = f"{user_id}:{content}:{datetime.now().isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Extract text content from a cognee search result.

        Handles multiple result formats: plain strings, dicts with
        text/content/memory keys, or objects with search_result attribute.

        Args:
            result: A single search result item.

        Returns:
            Extracted text string.
        """
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("text", "content", "memory"):
                if key in result:
                    return str(result[key])
            return str(result)
        if hasattr(result, "search_result"):
            sr = result.search_result
            if isinstance(sr, str):
                return sr
            if isinstance(sr, list) and sr:
                item = sr[0]
                if isinstance(item, dict):
                    return item.get("text", str(item))
                return str(item)
            return str(sr)
        return str(result)

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

        Stores the content in the user's cognee dataset. The knowledge graph
        is NOT automatically updated — call cognify() after a batch of writes.

        Args:
            content: The factual content to store.
            user_id: User identifier.
            metadata: Optional metadata (stored as context in future versions).
            memory_type: Type of memory (default: FACT).
            importance: Importance score (0.0 to 2.0, default: 1.0).
            tags: Optional tags for categorization.

        Returns:
            Memory ID string.
        """
        dataset = self._user_dataset(user_id)
        memory_id = self._generate_memory_id(content, user_id)

        try:
            await self._client.add(content, dataset)
            self._pending_datasets.add(dataset)

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

        Stores the content in the user's cognee dataset. The run_id is tracked
        but cognee does not natively separate runs — all user data goes to
        the same dataset and knowledge graph.

        Args:
            content: The episodic content to store.
            user_id: User identifier.
            run_id: Conversation run identifier.
            metadata: Optional metadata.
            memory_type: Type of memory (default: CONVERSATION).
            importance: Importance score (0.0 to 2.0, default: 1.0).
            tags: Optional tags for categorization.
            emotional_valence: Emotional tone (-1.0 to +1.0).

        Returns:
            Memory ID string.
        """
        dataset = self._user_dataset(user_id)
        memory_id = self._generate_memory_id(content, user_id)

        try:
            await self._client.add(content, dataset)
            self._pending_datasets.add(dataset)

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

    async def cognify(self) -> None:
        """Build/update the knowledge graph from all pending data.

        This is a potentially long-running operation that processes raw text
        into a structured knowledge graph. Call it periodically or after a
        batch of writes, NOT after every single write.

        After cognify, GRAPH_COMPLETION search becomes available for richer
        results that leverage entity relationships.
        """
        async with self._cognify_lock:
            pending = list(self._pending_datasets)
            if not pending:
                return
            try:
                await self._client.cognify(datasets=pending)
                for dataset in pending:
                    self._pending_datasets.discard(dataset)
                logger.info("Cognify completed successfully for {} datasets", len(pending))
            except Exception as e:
                logger.error("Cognify failed: {}", e)
                raise

    async def search(
        self,
        query: str,
        user_id: int,
        run_id: str | None = None,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryFact]:
        """Search memories using vector-based search.

        Uses cognee's chunk-level search for fast results without requiring
        a prior cognify() call.

        Args:
            query: Search query (semantic search).
            user_id: User identifier.
            run_id: Optional run_id filter (not natively supported by cognee).
            limit: Maximum number of results (default: 5).
            filters: Optional additional filters (reserved for future use).

        Returns:
            List of matching MemoryFact objects.
        """
        user_dataset = self._user_dataset(user_id)
        try:
            results = await self._client.search(query_text=query, datasets=[user_dataset])

            memories: list[MemoryFact] = []
            for result in results[:limit]:
                text = self._extract_text(result)
                memory = MemoryFact(
                    content=text,
                    user_id=user_id,
                    memory_type=MemoryType.TEXT,
                    memory_category=MemoryCategory.SEMANTIC,
                    fact_id=self._generate_memory_id(text, user_id),
                    timestamp=datetime.now(),
                )
                memories.append(memory)

            logger.debug(
                "Search for '{}' returned {} memories for user {}",
                query,
                len(memories),
                user_id,
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

        Uses a broad search query to retrieve stored content.

        Args:
            user_id: User identifier.
            run_id: Optional run_id filter.
            limit: Maximum number of results (default: 100).

        Returns:
            List of MemoryFact objects.
        """
        user_dataset = self._user_dataset(user_id)
        try:
            results = await self._client.search(query_text="*", datasets=[user_dataset])

            memories: list[MemoryFact] = []
            for result in results[:limit]:
                text = self._extract_text(result)
                memory = MemoryFact(
                    content=text,
                    user_id=user_id,
                    memory_type=MemoryType.TEXT,
                    memory_category=MemoryCategory.SEMANTIC,
                    fact_id=self._generate_memory_id(text, user_id),
                    timestamp=datetime.now(),
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
            run_id: Optional run_id to scope deletion.
        """
        dataset = self._user_dataset(user_id)
        try:
            await self._client.delete_dataset(dataset)
            self._pending_datasets.discard(dataset)
            logger.info(
                "Deleted memories for user {} (run_id={})",
                user_id,
                run_id,
            )
        except Exception as e:
            logger.error("Failed to delete memories for user {}: {}", user_id, e)
            raise


# Global instance for dependency injection
_memory_service: CogneeMemoryService | None = None


def get_memory_service() -> CogneeMemoryService:
    """Get or create global memory service instance.

    Returns:
        CogneeMemoryService instance.

    Raises:
        RuntimeError: If service cannot be initialized.
    """
    global _memory_service
    if _memory_service is None:
        _memory_service = CogneeMemoryService()
    return _memory_service


def set_memory_service(service: CogneeMemoryService | None) -> None:
    """Set global memory service instance (useful for testing).

    Args:
        service: CogneeMemoryService instance or None to reset.
    """
    global _memory_service
    _memory_service = service
