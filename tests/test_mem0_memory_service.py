"""Tests for Mem0 Memory Service.

All mem0 AsyncMemory interactions are mocked — no real network calls.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.memory.mem0_service import (
    _EMOTIONAL_KEYWORDS,
    _SESSION_KEYWORDS,
    RUSSIAN_EXTRACTION_PROMPT,
    Mem0MemoryService,
    get_memory_service,
    set_memory_service,
)
from bot.memory.models import MemoryCategory, MemoryType

# ---------------------------------------------------------------------------
# Helpers / fixtures shared across test classes
# ---------------------------------------------------------------------------


def _make_mock_client() -> AsyncMock:
    """Return a fresh AsyncMock that looks like a Mem0ClientProtocol."""
    client = AsyncMock()
    client.add = AsyncMock(return_value={"results": []})
    client.search = AsyncMock(return_value={"results": []})
    client.get_all = AsyncMock(return_value={"results": []})
    client.delete_all = AsyncMock(return_value={"results": []})
    return client


def _mem0_result(
    memory_id: str = "abc123",
    memory: str = "Test memory",
    memory_type: str = MemoryType.FACT.value,
    memory_category: str = MemoryCategory.SEMANTIC.value,
    importance: float = 1.0,
    tags: list[str] | None = None,
    score: float = 0.9,
    created_at: str | None = None,
) -> dict:
    """Build a realistic mem0 search/get_all result entry."""
    return {
        "id": memory_id,
        "memory": memory,
        "score": score,
        "created_at": created_at or "2024-01-15T10:00:00+00:00",
        "metadata": {
            "memory_type": memory_type,
            "memory_category": memory_category,
            "importance": importance,
            "tags": tags or [],
        },
    }


# ---------------------------------------------------------------------------
# Unit tests — constructor & config
# ---------------------------------------------------------------------------


class TestMem0MemoryServiceInit:
    def test_init_with_mock_client(self) -> None:
        client = _make_mock_client()
        service = Mem0MemoryService(client=client)
        assert service._client is client

    def test_mem0_not_available_raises(self) -> None:
        with (
            patch("bot.memory.mem0_service._mem0_available", False),
            pytest.raises(RuntimeError, match="mem0 package is not installed"),
        ):
            Mem0MemoryService()

    def test_user_key_formatting(self) -> None:
        client = _make_mock_client()
        service = Mem0MemoryService(client=client)
        assert service._user_key(12345) == "tg_user_12345"
        assert service._user_key(0) == "tg_user_0"
        assert service._user_key(999999) == "tg_user_999999"

    def test_generate_memory_id_returns_hex_string(self) -> None:
        memory_id = Mem0MemoryService._generate_memory_id("some content", 42)
        assert isinstance(memory_id, str)
        assert len(memory_id) == 16
        int(memory_id, 16)  # must be valid hex

    def test_generate_memory_id_different_inputs_differ(self) -> None:
        id1 = Mem0MemoryService._generate_memory_id("content A", 1)
        id2 = Mem0MemoryService._generate_memory_id("content B", 1)
        # Different content should produce different IDs (astronomically likely)
        assert id1 != id2

    def test_build_config_without_api_key(self) -> None:
        """When no LLM key is configured, config only has extraction prompt and version."""
        client = _make_mock_client()
        service = Mem0MemoryService(client=client)

        with patch("bot.memory.mem0_service.settings") as mock_settings:
            mock_settings.llm_api_key = None
            mock_settings.llm_base_url = None
            mock_settings.llm_model = "gpt-4o"

            config = service._build_config()

        assert "custom_fact_extraction_prompt" in config
        assert config["version"] == "v1.1"
        assert "llm" not in config
        assert "embedder" not in config

    def test_build_config_with_api_key(self) -> None:
        client = _make_mock_client()
        service = Mem0MemoryService(client=client)

        with patch("bot.memory.mem0_service.settings") as mock_settings:
            mock_settings.llm_api_key = "sk-test"
            mock_settings.llm_base_url = None
            mock_settings.llm_model = "gpt-4o"
            mock_settings.embedder_model = "text-embedding-3-small"
            mock_settings.mem0_supabase_connection_string = None

            config = service._build_config()

        assert config["llm"]["provider"] == "openai"
        assert config["llm"]["config"]["api_key"] == "sk-test"
        assert config["llm"]["config"]["model"] == "gpt-4o"
        assert config["embedder"]["provider"] == "openai"
        assert config["embedder"]["config"]["model"] == "text-embedding-3-small"
        assert "vector_store" not in config

    def test_build_config_with_base_url(self) -> None:
        client = _make_mock_client()
        service = Mem0MemoryService(client=client)

        with patch("bot.memory.mem0_service.settings") as mock_settings:
            mock_settings.llm_api_key = "sk-test"
            mock_settings.llm_base_url = "https://custom.endpoint/v1"
            mock_settings.llm_model = "kimi"
            mock_settings.embedder_model = "text-embedding-3-small"
            mock_settings.mem0_supabase_connection_string = "postgresql://user:pass@host/db"

            config = service._build_config()

        assert config["llm"]["config"]["openai_base_url"] == "https://custom.endpoint/v1"
        assert config["embedder"]["config"]["openai_base_url"] == "https://custom.endpoint/v1"
        assert config["vector_store"]["provider"] == "supabase"
        assert (
            config["vector_store"]["config"]["connection_string"]
            == "postgresql://user:pass@host/db"
        )


# ---------------------------------------------------------------------------
# Unit tests — TTL classification
# ---------------------------------------------------------------------------


class TestClassifyTtl:
    @pytest.fixture
    def service(self) -> Mem0MemoryService:
        return Mem0MemoryService(client=_make_mock_client())

    @pytest.mark.parametrize(
        "content",
        [
            "Пользователь устал",
            "Грустно сегодня",
            "Очень рад этому",
            "Волнуюсь перед встречей",
            "Настроение плохое",
            "Стресс на работе",
        ],
    )
    def test_emotional_content_gets_7_day_ttl(
        self, service: Mem0MemoryService, content: str
    ) -> None:
        expiry = service._classify_ttl(content, MemoryType.FACT)
        assert expiry is not None
        delta = expiry - datetime.now(tz=UTC)
        # Allow ±5 seconds tolerance
        assert abs(delta.total_seconds() - 7 * 86400) < 5

    def test_mood_state_type_gets_7_day_ttl_regardless_of_keywords(
        self, service: Mem0MemoryService
    ) -> None:
        expiry = service._classify_ttl("Нейтральный контент", MemoryType.MOOD_STATE)
        assert expiry is not None
        delta = expiry - datetime.now(tz=UTC)
        assert abs(delta.total_seconds() - 7 * 86400) < 5

    @pytest.mark.parametrize(
        "content",
        [
            "Сегодня был хороший день",
            "Сейчас читаю книгу",
            "Только что поел",
            "Вчера ходил в кино",
            "Завтра встреча",
        ],
    )
    def test_session_content_gets_30_day_ttl(
        self, service: Mem0MemoryService, content: str
    ) -> None:
        expiry = service._classify_ttl(content, MemoryType.FACT)
        assert expiry is not None
        delta = expiry - datetime.now(tz=UTC)
        assert abs(delta.total_seconds() - 30 * 86400) < 5

    def test_identity_content_has_no_expiry(self, service: Mem0MemoryService) -> None:
        expiry = service._classify_ttl("Работает программистом в Москве", MemoryType.IDENTITY)
        assert expiry is None

    def test_preference_content_has_no_expiry(self, service: Mem0MemoryService) -> None:
        expiry = service._classify_ttl("Любит пиццу и рок-музыку", MemoryType.LIKE)
        assert expiry is None

    def test_emotional_keyword_takes_priority_over_session(
        self, service: Mem0MemoryService
    ) -> None:
        # "устал сегодня" contains both emotional and session keywords
        expiry = service._classify_ttl("Устал сегодня после работы", MemoryType.FACT)
        assert expiry is not None
        delta = expiry - datetime.now(tz=UTC)
        # Should get 7 days (emotional), not 30 (session)
        assert abs(delta.total_seconds() - 7 * 86400) < 5


# ---------------------------------------------------------------------------
# Unit tests — write_factual
# ---------------------------------------------------------------------------


class TestWriteFactual:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        return _make_mock_client()

    @pytest.fixture
    def service(self, mock_client: AsyncMock) -> Mem0MemoryService:
        return Mem0MemoryService(client=mock_client)

    @pytest.mark.asyncio
    async def test_write_factual_success_returns_id(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.add.return_value = {"results": [{"id": "mem_abc123"}]}

        memory_id = await service.write_factual(content="Любит кофе", user_id=42)

        assert memory_id == "mem_abc123"

    @pytest.mark.asyncio
    async def test_write_factual_empty_results_uses_generated_id(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.add.return_value = {"results": []}

        memory_id = await service.write_factual(content="Работает инженером", user_id=42)

        assert isinstance(memory_id, str)
        assert len(memory_id) == 16

    @pytest.mark.asyncio
    async def test_write_factual_calls_add_with_correct_user_key(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        await service.write_factual(content="Факт о пользователе", user_id=123)

        call_kwargs = mock_client.add.call_args.kwargs
        assert call_kwargs["user_id"] == "tg_user_123"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Факт о пользователе"}]

    @pytest.mark.asyncio
    async def test_write_factual_includes_memory_type_in_metadata(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        await service.write_factual(
            content="Хочет стать предпринимателем",
            user_id=7,
            memory_type=MemoryType.GOAL,
            importance=1.8,
        )

        call_kwargs = mock_client.add.call_args.kwargs
        assert call_kwargs["metadata"]["memory_type"] == MemoryType.GOAL.value
        assert call_kwargs["metadata"]["importance"] == 1.8

    @pytest.mark.asyncio
    async def test_write_factual_includes_tags_in_metadata(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        await service.write_factual(
            content="Интересуется Python",
            user_id=7,
            tags=["python", "programming"],
        )

        call_kwargs = mock_client.add.call_args.kwargs
        assert call_kwargs["metadata"]["tags"] == ["python", "programming"]

    @pytest.mark.asyncio
    async def test_write_factual_merges_extra_metadata(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        await service.write_factual(
            content="Факт",
            user_id=1,
            metadata={"source": "conversation", "episode": "ep_001"},
        )

        call_kwargs = mock_client.add.call_args.kwargs
        assert call_kwargs["metadata"]["source"] == "conversation"
        assert call_kwargs["metadata"]["episode"] == "ep_001"

    @pytest.mark.asyncio
    async def test_write_factual_sets_expiration_for_emotional_content(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        await service.write_factual(content="Пользователь устал", user_id=5)

        call_kwargs = mock_client.add.call_args.kwargs
        assert "expiration_date" in call_kwargs
        assert call_kwargs["expiration_date"] is not None

    @pytest.mark.asyncio
    async def test_write_factual_no_expiration_for_identity(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        await service.write_factual(
            content="Зовут Мария, живёт в Санкт-Петербурге",
            user_id=5,
            memory_type=MemoryType.IDENTITY,
        )

        call_kwargs = mock_client.add.call_args.kwargs
        assert "expiration_date" not in call_kwargs

    @pytest.mark.asyncio
    async def test_write_factual_error_propagates(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.add.side_effect = RuntimeError("Network error")

        with pytest.raises(RuntimeError, match="Network error"):
            await service.write_factual(content="Факт", user_id=1)


# ---------------------------------------------------------------------------
# Unit tests — write_episodic
# ---------------------------------------------------------------------------


class TestWriteEpisodic:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        return _make_mock_client()

    @pytest.fixture
    def service(self, mock_client: AsyncMock) -> Mem0MemoryService:
        return Mem0MemoryService(client=mock_client)

    @pytest.mark.asyncio
    async def test_write_episodic_success(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.add.return_value = {"results": [{"id": "ep_789"}]}

        memory_id = await service.write_episodic(
            content="Обсуждали планы на выходные",
            user_id=10,
            run_id="session_xyz",
        )

        assert memory_id == "ep_789"

    @pytest.mark.asyncio
    async def test_write_episodic_includes_run_id_in_metadata(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        await service.write_episodic(
            content="Разговор о программировании",
            user_id=20,
            run_id="run_abc",
            emotional_valence=0.7,
        )

        call_kwargs = mock_client.add.call_args.kwargs
        assert call_kwargs["metadata"]["run_id"] == "run_abc"
        assert call_kwargs["metadata"]["emotional_valence"] == 0.7

    @pytest.mark.asyncio
    async def test_write_episodic_default_memory_type_is_conversation(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        await service.write_episodic(
            content="Событие беседы",
            user_id=30,
            run_id="run_001",
        )

        call_kwargs = mock_client.add.call_args.kwargs
        assert call_kwargs["metadata"]["memory_type"] == MemoryType.CONVERSATION.value

    @pytest.mark.asyncio
    async def test_write_episodic_error_propagates(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.add.side_effect = ConnectionError("Timeout")

        with pytest.raises(ConnectionError, match="Timeout"):
            await service.write_episodic(
                content="Событие",
                user_id=1,
                run_id="run_fail",
            )


# ---------------------------------------------------------------------------
# Unit tests — cognify
# ---------------------------------------------------------------------------


class TestCognify:
    @pytest.mark.asyncio
    async def test_cognify_is_noop(self) -> None:
        """cognify() should complete without error and not call any client method."""
        client = _make_mock_client()
        service = Mem0MemoryService(client=client)

        await service.cognify()

        client.add.assert_not_called()
        client.search.assert_not_called()
        client.get_all.assert_not_called()
        client.delete_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_cognify_returns_none(self) -> None:
        service = Mem0MemoryService(client=_make_mock_client())
        result = await service.cognify()
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests — search
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        return _make_mock_client()

    @pytest.fixture
    def service(self, mock_client: AsyncMock) -> Mem0MemoryService:
        return Mem0MemoryService(client=mock_client)

    @pytest.mark.asyncio
    async def test_search_maps_results_to_memory_facts(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {
            "results": [
                _mem0_result(memory_id="id1", memory="Любит кофе"),
                _mem0_result(memory_id="id2", memory="Работает программистом"),
            ]
        }

        results = await service.search(query="профессия", user_id=99)

        assert len(results) == 2
        assert results[0].content == "Любит кофе"
        assert results[0].fact_id == "id1"
        assert results[0].user_id == 99
        assert results[1].content == "Работает программистом"

    @pytest.mark.asyncio
    async def test_search_calls_client_with_correct_args(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        await service.search(query="интересы", user_id=55, limit=3)

        mock_client.search.assert_called_once_with(query="интересы", user_id="tg_user_55", limit=3)

    @pytest.mark.asyncio
    async def test_search_maps_memory_type_from_metadata(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {
            "results": [_mem0_result(memory_type=MemoryType.GOAL.value)]
        }

        results = await service.search(query="цель", user_id=1)

        assert results[0].memory_type == MemoryType.GOAL

    @pytest.mark.asyncio
    async def test_search_maps_memory_category_from_metadata(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {
            "results": [_mem0_result(memory_category=MemoryCategory.EMOTIONAL.value)]
        }

        results = await service.search(query="эмоции", user_id=1)

        assert results[0].memory_category == MemoryCategory.EMOTIONAL

    @pytest.mark.asyncio
    async def test_search_invalid_memory_type_falls_back_to_fact(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {
            "results": [
                {
                    "id": "x",
                    "memory": "data",
                    "metadata": {"memory_type": "invalid_type"},
                }
            ]
        }

        results = await service.search(query="test", user_id=1)

        assert results[0].memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_search_invalid_memory_category_falls_back_to_semantic(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {
            "results": [
                {
                    "id": "x",
                    "memory": "data",
                    "metadata": {"memory_category": "bogus_cat"},
                }
            ]
        }

        results = await service.search(query="test", user_id=1)

        assert results[0].memory_category == MemoryCategory.SEMANTIC

    @pytest.mark.asyncio
    async def test_search_maps_importance_from_metadata(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {"results": [_mem0_result(importance=1.8)]}

        results = await service.search(query="важность", user_id=1)

        assert results[0].importance_score == 1.8

    @pytest.mark.asyncio
    async def test_search_falls_back_to_score_when_no_importance(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {
            "results": [{"id": "x", "memory": "data", "score": 0.75, "metadata": {}}]
        }

        results = await service.search(query="score", user_id=1)

        assert results[0].importance_score == 0.75

    @pytest.mark.asyncio
    async def test_search_maps_tags_from_metadata(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {"results": [_mem0_result(tags=["python", "work"])]}

        results = await service.search(query="теги", user_id=1)

        assert results[0].tags == ["python", "work"]

    @pytest.mark.asyncio
    async def test_search_parses_created_at_timestamp(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {
            "results": [_mem0_result(created_at="2024-06-01T12:00:00+00:00")]
        }

        results = await service.search(query="время", user_id=1)

        assert results[0].timestamp == datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_search_uses_now_when_no_created_at(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {
            "results": [{"id": "x", "memory": "data", "metadata": {}}]
        }

        results = await service.search(query="test", user_id=1)

        assert results[0].timestamp is not None

    @pytest.mark.asyncio
    async def test_search_empty_results(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = {"results": []}

        results = await service.search(query="ничего", user_id=1)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_error_returns_empty_list(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.search.side_effect = RuntimeError("Search failed")

        results = await service.search(query="test", user_id=1)

        assert results == []


# ---------------------------------------------------------------------------
# Unit tests — get_all_memories
# ---------------------------------------------------------------------------


class TestGetAllMemories:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        return _make_mock_client()

    @pytest.fixture
    def service(self, mock_client: AsyncMock) -> Mem0MemoryService:
        return Mem0MemoryService(client=mock_client)

    @pytest.mark.asyncio
    async def test_get_all_success(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.get_all.return_value = {
            "results": [
                _mem0_result(memory_id="m1", memory="Факт один"),
                _mem0_result(memory_id="m2", memory="Факт два"),
            ]
        }

        results = await service.get_all_memories(user_id=77)

        assert len(results) == 2
        assert results[0].content == "Факт один"
        assert results[1].content == "Факт два"

    @pytest.mark.asyncio
    async def test_get_all_calls_correct_user_key(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        await service.get_all_memories(user_id=88)

        mock_client.get_all.assert_called_once_with(user_id="tg_user_88")

    @pytest.mark.asyncio
    async def test_get_all_respects_limit(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.get_all.return_value = {
            "results": [_mem0_result(memory_id=f"m{i}", memory=f"Факт {i}") for i in range(10)]
        }

        results = await service.get_all_memories(user_id=1, limit=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_all_error_returns_empty_list(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.get_all.side_effect = RuntimeError("DB error")

        results = await service.get_all_memories(user_id=1)

        assert results == []

    @pytest.mark.asyncio
    async def test_get_all_invalid_type_falls_back_to_fact(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.get_all.return_value = {
            "results": [{"id": "x", "memory": "data", "metadata": {"memory_type": "????"}}]
        }

        results = await service.get_all_memories(user_id=1)

        assert results[0].memory_type == MemoryType.FACT


# ---------------------------------------------------------------------------
# Unit tests — delete_user_memories
# ---------------------------------------------------------------------------


class TestDeleteUserMemories:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        return _make_mock_client()

    @pytest.fixture
    def service(self, mock_client: AsyncMock) -> Mem0MemoryService:
        return Mem0MemoryService(client=mock_client)

    @pytest.mark.asyncio
    async def test_delete_success(self, service: Mem0MemoryService, mock_client: AsyncMock) -> None:
        await service.delete_user_memories(user_id=5)

        mock_client.delete_all.assert_called_once_with(user_id="tg_user_5")

    @pytest.mark.asyncio
    async def test_delete_error_propagates(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        mock_client.delete_all.side_effect = RuntimeError("Delete failed")

        with pytest.raises(RuntimeError, match="Delete failed"):
            await service.delete_user_memories(user_id=5)

    @pytest.mark.asyncio
    async def test_delete_with_run_id_still_deletes_all(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        """run_id is metadata-only; mem0 has no run-scoped delete, so all are deleted."""
        await service.delete_user_memories(user_id=10, run_id="run_xyz")

        mock_client.delete_all.assert_called_once_with(user_id="tg_user_10")


# ---------------------------------------------------------------------------
# Integration patterns
# ---------------------------------------------------------------------------


class TestMem0WorkflowPatterns:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        return _make_mock_client()

    @pytest.fixture
    def service(self, mock_client: AsyncMock) -> Mem0MemoryService:
        return Mem0MemoryService(client=mock_client)

    @pytest.mark.asyncio
    async def test_write_then_search_workflow(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        """Demonstrate typical write → search workflow."""
        mock_client.add.return_value = {"results": [{"id": "f001"}]}
        mock_client.search.return_value = {
            "results": [_mem0_result(memory_id="f001", memory="Любит рок-музыку")]
        }

        fact_id = await service.write_factual(
            content="Пользователь любит рок-музыку",
            user_id=100,
            tags=["music"],
        )
        assert fact_id == "f001"

        results = await service.search(query="музыка", user_id=100)

        assert len(results) == 1
        assert "рок" in results[0].content

    @pytest.mark.asyncio
    async def test_user_isolation_via_key(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        """All writes and searches use the user-scoped key."""
        await service.write_factual(content="Факт A", user_id=1)
        await service.write_factual(content="Факт B", user_id=2)
        await service.search(query="что-то", user_id=1)

        add_calls = mock_client.add.call_args_list
        assert add_calls[0].kwargs["user_id"] == "tg_user_1"
        assert add_calls[1].kwargs["user_id"] == "tg_user_2"

        search_call = mock_client.search.call_args
        assert search_call.kwargs["user_id"] == "tg_user_1"

    @pytest.mark.asyncio
    async def test_cognify_noop_in_workflow(
        self, service: Mem0MemoryService, mock_client: AsyncMock
    ) -> None:
        """cognify() in a workflow doesn't break anything."""
        await service.write_factual(content="Факт", user_id=1)
        await service.cognify()  # should be a no-op
        await service.search(query="факт", user_id=1)

        mock_client.add.assert_called_once()
        mock_client.search.assert_called_once()


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------


class TestSingletonManagement:
    def setup_method(self) -> None:
        set_memory_service(None)

    def teardown_method(self) -> None:
        set_memory_service(None)

    def test_set_and_get_returns_same_instance(self) -> None:
        client = _make_mock_client()
        custom_service = Mem0MemoryService(client=client)

        set_memory_service(custom_service)

        assert get_memory_service() is custom_service

    def test_set_to_none_clears_instance(self) -> None:
        client = _make_mock_client()
        service = Mem0MemoryService(client=client)
        set_memory_service(service)

        set_memory_service(None)

        # get_memory_service() would try to build a real instance — just verify None was set
        with patch("bot.memory.mem0_service.Mem0MemoryService") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_memory_service()
            mock_cls.assert_called_once()

    def test_set_overrides_existing_instance(self) -> None:
        service_a = Mem0MemoryService(client=_make_mock_client())
        service_b = Mem0MemoryService(client=_make_mock_client())

        set_memory_service(service_a)
        set_memory_service(service_b)

        assert get_memory_service() is service_b


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_russian_extraction_prompt_exists(self) -> None:
        assert isinstance(RUSSIAN_EXTRACTION_PROMPT, str)
        assert len(RUSSIAN_EXTRACTION_PROMPT) > 0

    def test_russian_extraction_prompt_contains_personal_info_category(self) -> None:
        assert "Личная информация" in RUSSIAN_EXTRACTION_PROMPT

    def test_russian_extraction_prompt_contains_preferences_category(self) -> None:
        assert "Предпочтения" in RUSSIAN_EXTRACTION_PROMPT

    def test_russian_extraction_prompt_contains_emotional_category(self) -> None:
        assert "Эмоциональное состояние" in RUSSIAN_EXTRACTION_PROMPT

    def test_russian_extraction_prompt_contains_plans_category(self) -> None:
        assert "Планы и намерения" in RUSSIAN_EXTRACTION_PROMPT

    def test_russian_extraction_prompt_contains_relationships_category(self) -> None:
        assert "Отношения" in RUSSIAN_EXTRACTION_PROMPT

    def test_russian_extraction_prompt_contains_habits_category(self) -> None:
        assert "Привычки и рутина" in RUSSIAN_EXTRACTION_PROMPT

    def test_russian_extraction_prompt_has_json_output_instruction(self) -> None:
        assert '"facts"' in RUSSIAN_EXTRACTION_PROMPT

    def test_russian_extraction_prompt_has_examples(self) -> None:
        assert "Алексей" in RUSSIAN_EXTRACTION_PROMPT
        assert "Москве" in RUSSIAN_EXTRACTION_PROMPT

    def test_emotional_keywords_is_non_empty_frozenset(self) -> None:
        assert isinstance(_EMOTIONAL_KEYWORDS, frozenset)
        assert len(_EMOTIONAL_KEYWORDS) > 0
        assert "устал" in _EMOTIONAL_KEYWORDS
        assert "стресс" in _EMOTIONAL_KEYWORDS

    def test_session_keywords_is_non_empty_frozenset(self) -> None:
        assert isinstance(_SESSION_KEYWORDS, frozenset)
        assert len(_SESSION_KEYWORDS) > 0
        assert "сегодня" in _SESSION_KEYWORDS
        assert "сейчас" in _SESSION_KEYWORDS
