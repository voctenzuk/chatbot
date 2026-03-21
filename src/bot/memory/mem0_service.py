"""Mem0 Memory Service for automatic fact extraction and vector search.

Replaces CogneeMemoryService with mem0's LLM-powered pipeline:
    1. LLM extracts facts from conversation using a Russian-language prompt
    2. LLM resolves conflicts with existing memories (ADD/UPDATE/DELETE)
    3. Facts stored in the configured vector store with optional TTL

Unlike Cognee, mem0 handles fact extraction automatically — there is no
separate cognify() step. The cognify() method is kept as a no-op for
protocol compatibility.

Configuration (via environment variables or .env file):
    LLM_API_KEY: API key for the LLM provider
    LLM_BASE_URL: Custom LLM endpoint URL (optional)
    LLM_MODEL: LLM model name (optional)
"""

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from bot.config import settings
from bot.memory.models import MemoryCategory, MemoryFact, MemoryType

# Import mem0 — may not be available in all environments
try:
    from mem0 import AsyncMemory

    _mem0_available = True
except ImportError:
    _mem0_available = False
    AsyncMemory = None

# ---------------------------------------------------------------------------
# Russian extraction prompt
# ---------------------------------------------------------------------------

RUSSIAN_EXTRACTION_PROMPT = """
Ты — ассистент по извлечению фактов из разговора.
Извлеки КЛЮЧЕВЫЕ факты о пользователе. Категории:
- Личная информация (имя, город, возраст, работа)
- Предпочтения (любит/не любит, интересы, хобби)
- Эмоциональное состояние (настроение, стресс, радость)
- Планы и намерения (собирается, хочет, мечтает)
- Отношения (друзья, семья, коллеги)
- Привычки и рутина (обычно делает, расписание)

Input: Привет
Output: {"facts": []}

Input: Меня зовут Алексей, я работаю программистом в Москве
Output: {"facts": ["Имя пользователя: Алексей", "Работает программистом", "Живёт в Москве"]}

Input: Устал сегодня, был тяжёлый день на работе
Output: {"facts": ["Пользователь устал", "Был тяжёлый рабочий день"]}

Верни JSON с ключом "facts". Сохраняй язык фактов — русский.
"""

# ---------------------------------------------------------------------------
# TTL keyword sets
# ---------------------------------------------------------------------------

_EMOTIONAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "устал",
        "грустн",
        "злой",
        "раздражен",
        "счастлив",
        "рад",
        "волнуюсь",
        "переживаю",
        "настроение",
        "стресс",
    }
)

_SESSION_KEYWORDS: frozenset[str] = frozenset(
    {"сегодня", "сейчас", "только что", "вчера", "завтра"}
)

# ---------------------------------------------------------------------------
# Protocol for the mem0 AsyncMemory client (enables testing without real mem0)
# ---------------------------------------------------------------------------


@runtime_checkable
class Mem0ClientProtocol(Protocol):
    """Protocol for the mem0 AsyncMemory client to enable testing."""

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> dict[str, Any]: ...

    async def get_all(
        self,
        user_id: str,
    ) -> dict[str, Any]: ...

    async def delete_all(
        self,
        user_id: str,
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class Mem0MemoryService:
    """Memory service using mem0 for automatic fact extraction and vector search.

    Pipeline per write_factual() / write_episodic():
      1. LLM extracts facts from the content using the Russian prompt
      2. LLM resolves conflicts with existing memories (ADD/UPDATE/DELETE)
      3. Facts stored in the configured vector store with optional TTL

    Unlike Cognee, mem0 handles knowledge extraction automatically.
    cognify() is a no-op kept for MemoryPort protocol compatibility.

    Data isolation is achieved via per-user user_id keys (tg_user_{user_id}).
    """

    def __init__(self, client: Mem0ClientProtocol | None = None) -> None:
        """Initialize the mem0 memory service.

        Args:
            client: Optional pre-configured mem0 client (for testing/mocking).

        Raises:
            RuntimeError: If mem0 package is not installed and no client provided.
        """
        self._client: Mem0ClientProtocol

        if client is not None:
            self._client = client
            logger.info("Mem0MemoryService initialized with custom client")
            return

        if not _mem0_available:
            raise RuntimeError("mem0 package is not installed. Install it with: pip install mem0ai")

        config = self._build_config()
        self._client = AsyncMemory.from_config(config)  # type: ignore[union-attr]
        logger.info("Mem0MemoryService initialized")

    def _build_config(self) -> dict[str, Any]:
        """Build mem0 config dict from application settings."""
        config: dict[str, Any] = {
            "custom_fact_extraction_prompt": RUSSIAN_EXTRACTION_PROMPT,
            "version": "v1.1",
        }

        if settings.llm_api_key:
            llm_cfg: dict[str, Any] = {
                "model": settings.llm_model,
                "temperature": 0.2,
                "max_tokens": 1500,
            }
            if settings.llm_base_url:
                llm_cfg["openai_base_url"] = settings.llm_base_url
            llm_cfg["api_key"] = settings.llm_api_key
            config["llm"] = {"provider": "openai", "config": llm_cfg}

            embedder_cfg: dict[str, Any] = {"model": settings.embedder_model}
            if settings.llm_base_url:
                embedder_cfg["openai_base_url"] = settings.llm_base_url
            embedder_cfg["api_key"] = settings.llm_api_key
            config["embedder"] = {"provider": "openai", "config": embedder_cfg}

        # Vector store: use Supabase pgvector if connection string is configured
        if settings.mem0_supabase_connection_string:
            config["vector_store"] = {
                "provider": "supabase",
                "config": {
                    "connection_string": settings.mem0_supabase_connection_string,
                    "collection_name": "memories",
                },
            }

        return config

    @staticmethod
    def _user_key(user_id: int) -> str:
        """Format user key for mem0 data isolation."""
        return f"tg_user_{user_id}"

    @staticmethod
    def _generate_memory_id(content: str, user_id: int) -> str:
        """Generate a deterministic memory ID from content and user.

        Args:
            content: Memory content.
            user_id: User identifier.

        Returns:
            Short hex hash string.
        """
        hash_input = f"{user_id}:{content}:{datetime.now(tz=UTC).isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _classify_ttl(self, content: str, memory_type: MemoryType) -> datetime | None:
        """Classify TTL based on content keywords and memory type.

        Args:
            content: Memory content to analyse.
            memory_type: Type of memory being stored.

        Returns:
            Expiration datetime or None for permanent storage.
        """
        content_lower = content.lower()

        # Emotional state → 7 days
        if memory_type == MemoryType.MOOD_STATE or any(
            kw in content_lower for kw in _EMOTIONAL_KEYWORDS
        ):
            return datetime.now(tz=UTC) + timedelta(days=7)

        # Session context → 30 days
        if any(kw in content_lower for kw in _SESSION_KEYWORDS):
            return datetime.now(tz=UTC) + timedelta(days=30)

        # Identity / preference → no expiry
        return None

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

        mem0 automatically extracts and deduplicates facts via its LLM pipeline.

        Args:
            content: The factual content to store.
            user_id: User identifier.
            metadata: Optional metadata merged into mem0 metadata.
            memory_type: Type of memory (default: FACT).
            importance: Importance score (0.0 to 2.0, default: 1.0).
            tags: Optional tags for categorization.

        Returns:
            Memory ID string (first result ID or generated hash).
        """
        user_key = self._user_key(user_id)
        mem_metadata: dict[str, Any] = {
            "memory_type": memory_type.value,
            "importance": importance,
        }
        if tags:
            mem_metadata["tags"] = tags
        if metadata:
            mem_metadata.update(metadata)

        expiration = self._classify_ttl(content, memory_type)

        kwargs: dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
            "user_id": user_key,
            "metadata": mem_metadata,
        }
        if expiration is not None:
            kwargs["expiration_date"] = expiration.isoformat()

        try:
            result = await self._client.add(**kwargs)
            results = result.get("results", [])
            memory_id = (
                results[0].get("id", self._generate_memory_id(content, user_id))
                if results
                else self._generate_memory_id(content, user_id)
            )
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

        Args:
            content: The episodic content to store.
            user_id: User identifier.
            run_id: Conversation run identifier (stored in metadata).
            metadata: Optional extra metadata.
            memory_type: Type of memory (default: CONVERSATION).
            importance: Importance score (0.0 to 2.0, default: 1.0).
            tags: Optional tags for categorization.
            emotional_valence: Emotional tone (-1.0 to +1.0).

        Returns:
            Memory ID string.
        """
        ep_metadata: dict[str, Any] = {"run_id": run_id, "emotional_valence": emotional_valence}
        if metadata:
            ep_metadata.update(metadata)

        try:
            memory_id = await self.write_factual(
                content=content,
                user_id=user_id,
                metadata=ep_metadata,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
            )
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
        """No-op — mem0 extracts knowledge automatically during add().

        Kept for MemoryPort protocol compatibility.
        """
        logger.debug("cognify() called on Mem0MemoryService — no-op (mem0 extracts automatically)")

    @staticmethod
    def _raw_to_memory_fact(r: dict[str, Any], user_id: int) -> MemoryFact:
        """Convert a raw mem0 result dict to a MemoryFact."""
        meta = r.get("metadata", {})
        try:
            mem_type = MemoryType(meta.get("memory_type", MemoryType.FACT.value))
        except ValueError:
            mem_type = MemoryType.FACT
        try:
            mem_category = MemoryCategory(
                meta.get("memory_category", MemoryCategory.SEMANTIC.value)
            )
        except ValueError:
            mem_category = MemoryCategory.SEMANTIC

        created_at = (
            datetime.fromisoformat(r["created_at"]) if "created_at" in r else datetime.now(tz=UTC)
        )

        return MemoryFact(
            content=r.get("memory", ""),
            user_id=user_id,
            memory_type=mem_type,
            memory_category=mem_category,
            fact_id=r.get("id", ""),
            importance_score=meta.get("importance", r.get("score", 1.0)),
            tags=meta.get("tags", []),
            timestamp=created_at,
        )

    async def search(
        self,
        query: str,
        user_id: int,
        run_id: str | None = None,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryFact]:
        """Search memories using vector similarity.

        Args:
            query: Search query (semantic search).
            user_id: User identifier.
            run_id: Optional run_id filter (not used by mem0 natively).
            limit: Maximum number of results (default: 5).
            filters: Optional additional filters (reserved for future use).

        Returns:
            List of matching MemoryFact objects, or empty list on error.
        """
        user_key = self._user_key(user_id)
        try:
            result = await self._client.search(query=query, user_id=user_key, limit=limit)
            memories = [self._raw_to_memory_fact(r, user_id) for r in result.get("results", [])]
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

        Args:
            user_id: User identifier.
            run_id: Optional run_id filter (not natively supported by mem0).
            limit: Maximum number of results (default: 100).

        Returns:
            List of MemoryFact objects, or empty list on error.
        """
        user_key = self._user_key(user_id)
        try:
            result = await self._client.get_all(user_id=user_key)
            return [self._raw_to_memory_fact(r, user_id) for r in result.get("results", [])[:limit]]
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
            run_id: Optional run_id to scope deletion (not natively supported by mem0).

        Raises:
            Exception: If the delete operation fails.
        """
        user_key = self._user_key(user_id)
        try:
            await self._client.delete_all(user_id=user_key)
            logger.info("Deleted memories for user {} (run_id={})", user_id, run_id)
        except Exception as e:
            logger.error("Failed to delete memories for user {}: {}", user_id, e)
            raise


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_service_holder: dict[str, Mem0MemoryService] = {}


def get_memory_service() -> Mem0MemoryService:
    """Get or create the global Mem0MemoryService instance.

    Returns:
        Mem0MemoryService instance.

    Raises:
        RuntimeError: If the service cannot be initialized (mem0 not installed).
    """
    if "instance" not in _service_holder:
        _service_holder["instance"] = Mem0MemoryService()
    return _service_holder["instance"]


def set_memory_service(service: Mem0MemoryService | None) -> None:
    """Set the global Mem0MemoryService instance (useful for testing).

    Args:
        service: Mem0MemoryService instance or None to reset.
    """
    if service is None:
        _service_holder.pop("instance", None)
    else:
        _service_holder["instance"] = service
