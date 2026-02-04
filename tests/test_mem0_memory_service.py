"""Tests for Mem0 Memory Service wrapper.

These tests use mocks to avoid real network calls.
All Mem0 API interactions are mocked using unittest.mock.MagicMock.
"""

import pytest
from unittest.mock import MagicMock, patch

from bot.services.mem0_memory_service import (
    Mem0MemoryService,
    get_memory_service,
    set_memory_service,
)
from bot.services.memory_models import MemoryType, MemoryCategory


class TestMem0MemoryServiceUnit:
    """Unit tests for Mem0MemoryService using mocks."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Mem0 client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def service(self, mock_client):
        """Create service with mock client."""
        return Mem0MemoryService(client=mock_client)

    def test_init_with_mock_client(self, mock_client):
        """Test initialization with mock client."""
        service = Mem0MemoryService(client=mock_client)
        assert service._client is mock_client
        assert service._is_mock is True

    def test_init_without_client_or_config(self):
        """Test initialization fails without client or config."""
        with patch.object(Mem0MemoryService, "__init__", lambda s, **kw: None):
            # Skip the actual init to test the check separately
            pass

        # Test that RuntimeError is raised when no API key or host
        with pytest.raises(RuntimeError, match="API key is required"):
            with patch.dict("os.environ", {}, clear=True):
                with patch("bot.services.mem0_memory_service.settings") as mock_settings:
                    mock_settings.mem0_api_key = None
                    mock_settings.mem0_project_id = None
                    Mem0MemoryService()

    def test_format_user_id(self, service):
        """Test user ID formatting."""
        assert service._format_user_id(12345) == "tg_user_12345"
        assert service._format_user_id(0) == "tg_user_0"
        assert service._format_user_id(999999) == "tg_user_999999"

    @pytest.mark.asyncio
    async def test_write_factual_success(self, service, mock_client):
        """Test writing factual memory successfully."""
        # Setup mock response
        mock_client.add.return_value = {
            "results": [{"id": "mem_123", "memory": "User likes pizza"}]
        }

        memory_id = await service.write_factual(
            content="User likes pizza",
            user_id=123,
            metadata={"source": "conversation"},
            importance=1.5,
            tags=["food", "preference"],
        )

        assert memory_id == "mem_123"
        mock_client.add.assert_called_once()
        call_args = mock_client.add.call_args
        assert call_args.kwargs["user_id"] == "tg_user_123"
        assert call_args.kwargs["metadata"]["is_factual"] is True
        assert call_args.kwargs["metadata"]["importance_score"] == 1.5
        assert call_args.kwargs["metadata"]["tags"] == ["food", "preference"]

    @pytest.mark.asyncio
    async def test_write_factual_default_values(self, service, mock_client):
        """Test writing factual memory with default values."""
        mock_client.add.return_value = {
            "results": [{"id": "mem_456", "memory": "User is a developer"}]
        }

        memory_id = await service.write_factual(
            content="User is a developer",
            user_id=456,
        )

        assert memory_id == "mem_456"
        call_args = mock_client.add.call_args
        assert call_args.kwargs["metadata"]["memory_type"] == MemoryType.FACT.value
        assert call_args.kwargs["metadata"]["memory_category"] == MemoryCategory.SEMANTIC.value
        assert call_args.kwargs["metadata"]["importance_score"] == 1.0
        assert call_args.kwargs["metadata"]["tags"] == []

    @pytest.mark.asyncio
    async def test_write_factual_empty_response(self, service, mock_client):
        """Test writing factual memory with empty response."""
        mock_client.add.return_value = {"results": []}

        memory_id = await service.write_factual(
            content="Some fact",
            user_id=123,
        )

        assert memory_id == ""

    @pytest.mark.asyncio
    async def test_write_factual_error(self, service, mock_client):
        """Test error handling when writing factual memory."""
        mock_client.add.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await service.write_factual(
                content="Some fact",
                user_id=123,
            )

    @pytest.mark.asyncio
    async def test_write_episodic_success(self, service, mock_client):
        """Test writing episodic memory successfully."""
        mock_client.add.return_value = {
            "results": [{"id": "mem_789", "memory": "User asked about Python"}]
        }

        memory_id = await service.write_episodic(
            content="User asked about Python async patterns",
            user_id=123,
            run_id="session_abc",
            metadata={"topic": "programming"},
            importance=1.2,
            tags=["python", "async"],
            emotional_valence=0.8,
        )

        assert memory_id == "mem_789"
        mock_client.add.assert_called_once()
        call_args = mock_client.add.call_args
        assert call_args.kwargs["user_id"] == "tg_user_123"
        assert call_args.kwargs["run_id"] == "session_abc"
        assert call_args.kwargs["metadata"]["is_episodic"] is True
        assert call_args.kwargs["metadata"]["emotional_valence"] == 0.8

    @pytest.mark.asyncio
    async def test_write_episodic_default_values(self, service, mock_client):
        """Test writing episodic memory with default values."""
        mock_client.add.return_value = {
            "results": [{"id": "mem_ep1", "memory": "Conversation event"}]
        }

        memory_id = await service.write_episodic(
            content="Conversation event",
            user_id=789,
            run_id="run_xyz",
        )

        assert memory_id == "mem_ep1"
        call_args = mock_client.add.call_args
        assert call_args.kwargs["metadata"]["memory_type"] == MemoryType.CONVERSATION.value
        assert call_args.kwargs["metadata"]["memory_category"] == MemoryCategory.EPISODIC.value
        assert call_args.kwargs["metadata"]["emotional_valence"] == 0.0

    @pytest.mark.asyncio
    async def test_write_episodic_error(self, service, mock_client):
        """Test error handling when writing episodic memory."""
        mock_client.add.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            await service.write_episodic(
                content="Some event",
                user_id=123,
                run_id="run_123",
            )

    @pytest.mark.asyncio
    async def test_search_success(self, service, mock_client):
        """Test searching memories successfully."""
        mock_client.search.return_value = {
            "results": [
                {
                    "id": "mem_1",
                    "memory": "User likes pizza",
                    "metadata": {
                        "memory_type": "fact",
                        "memory_category": "semantic",
                        "importance_score": 1.5,
                        "tags": ["food"],
                    },
                    "created_at": "2024-01-15T10:30:00Z",
                },
                {
                    "id": "mem_2",
                    "memory": "User enjoys hiking",
                    "metadata": {
                        "memory_type": "fact",
                        "memory_category": "semantic",
                    },
                    "created_at": "2024-01-14T09:00:00+00:00",
                },
            ]
        }

        results = await service.search(
            query="What does user like?",
            user_id=123,
            limit=5,
        )

        assert len(results) == 2
        assert results[0].fact_id == "mem_1"
        assert results[0].content == "User likes pizza"
        assert results[0].memory_type == MemoryType.FACT
        assert results[0].memory_category == MemoryCategory.SEMANTIC
        assert results[0].importance_score == 1.5
        assert results[0].tags == ["food"]
        mock_client.search.assert_called_once_with(
            query="What does user like?",
            user_id="tg_user_123",
            limit=5,
        )

    @pytest.mark.asyncio
    async def test_search_with_run_id(self, service, mock_client):
        """Test searching memories with run_id filter."""
        mock_client.search.return_value = {"results": []}

        await service.search(
            query="test",
            user_id=456,
            run_id="session_123",
            limit=3,
        )

        mock_client.search.assert_called_once_with(
            query="test",
            user_id="tg_user_456",
            run_id="session_123",
            limit=3,
        )

    @pytest.mark.asyncio
    async def test_search_with_filters(self, service, mock_client):
        """Test searching memories with additional filters."""
        mock_client.search.return_value = {"results": []}

        filters = {"importance_score": {"$gte": 1.0}}
        await service.search(
            query="test",
            user_id=123,
            filters=filters,
        )

        call_args = mock_client.search.call_args
        assert call_args.kwargs["filters"] == filters

    @pytest.mark.asyncio
    async def test_search_empty_results(self, service, mock_client):
        """Test searching with no results."""
        mock_client.search.return_value = {"results": []}

        results = await service.search(
            query="nonexistent",
            user_id=123,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self, service, mock_client):
        """Test that search errors return empty list instead of raising."""
        mock_client.search.side_effect = Exception("Search failed")

        results = await service.search(
            query="test",
            user_id=123,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_invalid_memory_type(self, service, mock_client):
        """Test search handles invalid memory types gracefully."""
        mock_client.search.return_value = {
            "results": [
                {
                    "id": "mem_1",
                    "memory": "Test",
                    "metadata": {
                        "memory_type": "invalid_type",
                        "memory_category": "invalid_category",
                    },
                    "created_at": "2024-01-15T10:30:00Z",
                },
            ]
        }

        results = await service.search(query="test", user_id=123)

        # Should default to TEXT and EPISODIC for invalid values
        assert results[0].memory_type == MemoryType.TEXT
        assert results[0].memory_category == MemoryCategory.EPISODIC

    @pytest.mark.asyncio
    async def test_get_all_memories_success(self, service, mock_client):
        """Test getting all memories successfully."""
        mock_client.get_all.return_value = {
            "results": [
                {
                    "id": "mem_1",
                    "memory": "Memory 1",
                    "metadata": {"memory_type": "fact"},
                    "created_at": "2024-01-15T10:30:00Z",
                },
                {
                    "id": "mem_2",
                    "memory": "Memory 2",
                    "metadata": {"memory_type": "conversation"},
                    "created_at": "2024-01-14T09:00:00Z",
                },
            ]
        }

        results = await service.get_all_memories(user_id=123, limit=10)

        assert len(results) == 2
        assert results[0].fact_id == "mem_1"
        assert results[1].fact_id == "mem_2"
        mock_client.get_all.assert_called_once_with(
            user_id="tg_user_123",
            limit=10,
        )

    @pytest.mark.asyncio
    async def test_get_all_memories_with_run_id(self, service, mock_client):
        """Test getting all memories filtered by run_id."""
        mock_client.get_all.return_value = {"results": []}

        await service.get_all_memories(user_id=123, run_id="session_1")

        mock_client.get_all.assert_called_once_with(
            user_id="tg_user_123",
            run_id="session_1",
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_get_all_memories_error_returns_empty(self, service, mock_client):
        """Test that get_all errors return empty list."""
        mock_client.get_all.side_effect = Exception("API Error")

        results = await service.get_all_memories(user_id=123)

        assert results == []

    @pytest.mark.asyncio
    async def test_delete_user_memories_success(self, service, mock_client):
        """Test deleting all user memories successfully."""
        mock_client.delete_all.return_value = None

        await service.delete_user_memories(user_id=123)

        mock_client.delete_all.assert_called_once_with(user_id="tg_user_123")

    @pytest.mark.asyncio
    async def test_delete_user_memories_with_run_id(self, service, mock_client):
        """Test deleting memories for specific run_id."""
        mock_client.delete_all.return_value = None

        await service.delete_user_memories(user_id=123, run_id="session_1")

        mock_client.delete_all.assert_called_once_with(
            user_id="tg_user_123",
            run_id="session_1",
        )

    @pytest.mark.asyncio
    async def test_delete_user_memories_error(self, service, mock_client):
        """Test error handling when deleting memories."""
        mock_client.delete_all.side_effect = Exception("Delete failed")

        with pytest.raises(Exception, match="Delete failed"):
            await service.delete_user_memories(user_id=123)


class TestMem0MemoryServiceIntegrationPatterns:
    """Integration pattern tests demonstrating usage."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Mem0 client."""
        return MagicMock()

    @pytest.fixture
    def service(self, mock_client):
        """Create service with mock client."""
        return Mem0MemoryService(client=mock_client)

    @pytest.mark.asyncio
    async def test_fact_and_episodic_workflow(self, service, mock_client):
        """Test combined workflow of factual and episodic memories."""
        # Setup mocks
        mock_client.add.side_effect = [
            {"results": [{"id": "fact_1"}]},  # First call: factual
            {"results": [{"id": "ep_1"}]},  # Second call: episodic
            {"results": [{"id": "ep_2"}]},  # Third call: episodic
        ]
        mock_client.search.return_value = {
            "results": [
                {
                    "id": "fact_1",
                    "memory": "User likes Python",
                    "metadata": {},
                    "created_at": "2024-01-15T10:30:00Z",
                },
                {
                    "id": "ep_1",
                    "memory": "Discussed async",
                    "metadata": {},
                    "created_at": "2024-01-15T10:35:00Z",
                },
            ]
        }

        # Store user preference (factual)
        fact_id = await service.write_factual(
            content="User likes Python programming",
            user_id=123,
            tags=["programming", "python"],
            importance=1.5,
        )
        assert fact_id == "fact_1"

        # Store conversation events (episodic)
        run_id = "session_2024_01_15"
        ep_id1 = await service.write_episodic(
            content="User and bot discussed Python async patterns",
            user_id=123,
            run_id=run_id,
            emotional_valence=0.6,
        )
        assert ep_id1 == "ep_1"

        ep_id2 = await service.write_episodic(
            content="User was excited about asyncio library",
            user_id=123,
            run_id=run_id,
            emotional_valence=0.9,
        )
        assert ep_id2 == "ep_2"

        # Search for relevant memories
        results = await service.search(
            query="What does user like about Python?",
            user_id=123,
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_conversation_session_isolation(self, service, mock_client):
        """Test that different conversation sessions are isolated."""
        mock_client.search.return_value = {"results": []}

        # Search within specific run
        await service.search(
            query="test",
            user_id=123,
            run_id="session_A",
        )

        # Verify run_id is passed
        assert mock_client.search.call_args.kwargs["run_id"] == "session_A"


class TestMem0MemoryServiceGlobalInstance:
    """Tests for global service instance management."""

    def setup_method(self):
        """Reset global instance before each test."""
        set_memory_service(None)

    def teardown_method(self):
        """Reset global instance after each test."""
        set_memory_service(None)

    def test_get_memory_service_creates_instance(self):
        """Test that get_memory_service creates instance when needed."""
        with patch("bot.services.mem0_memory_service.Mem0MemoryService") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            service = get_memory_service()

            mock_cls.assert_called_once()
            assert service is mock_instance

    def test_set_memory_service(self):
        """Test setting global instance."""
        mock_client = MagicMock()
        custom_service = Mem0MemoryService(client=mock_client)

        set_memory_service(custom_service)

        retrieved = get_memory_service()
        assert retrieved is custom_service

    def test_set_memory_service_to_none_resets(self):
        """Test resetting global instance to None."""
        mock_client = MagicMock()
        custom_service = Mem0MemoryService(client=mock_client)

        set_memory_service(custom_service)
        set_memory_service(None)

        with patch("bot.services.mem0_memory_service.Mem0MemoryService") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_memory_service()
            mock_cls.assert_called_once()  # Should create new instance


class TestMem0MemoryServiceConfiguration:
    """Tests for configuration and initialization."""

    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        mock_client = MagicMock()

        with patch("bot.services.mem0_memory_service.MEM0_AVAILABLE", True):
            with patch("bot.services.mem0_memory_service.MemoryClient", return_value=mock_client):
                with patch.dict(
                    "os.environ",
                    {
                        "MEM0_API_KEY": "test_key",
                        "MEM0_PROJECT_ID": "test_project",
                        "MEM0_ORG_ID": "test_org",
                        "MEM0_HOST": "http://localhost:8000",
                    },
                    clear=True,
                ):
                    service = Mem0MemoryService()

        assert service._api_key == "test_key"
        assert service._project_id == "test_project"
        assert service._org_id == "test_org"
        assert service._host == "http://localhost:8000"

    def test_init_with_explicit_params(self):
        """Test initialization with explicit parameters."""
        mock_client_cls = MagicMock()

        with patch("bot.services.mem0_memory_service.MEM0_AVAILABLE", True):
            with patch("bot.services.mem0_memory_service.MemoryClient", mock_client_cls):
                service = Mem0MemoryService(
                    api_key="explicit_key",
                    project_id="explicit_project",
                    org_id="explicit_org",
                    host="http://custom:8080",
                )

        assert service._api_key == "explicit_key"
        mock_client_cls.assert_called_once_with(
            api_key="explicit_key",
            project_id="explicit_project",
            org_id="explicit_org",
            host="http://custom:8080",
        )

    def test_mem0_not_available_raises(self):
        """Test that RuntimeError is raised when mem0 not installed."""
        with patch("bot.services.mem0_memory_service.MEM0_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="mem0 package is not installed"):
                Mem0MemoryService()


class TestDocumentationPresent:
    """Verify that documentation strings are present."""

    def test_pgvector_config_documented(self):
        """Verify pgvector configuration is documented in module."""
        import bot.services.mem0_memory_service as module

        docstring = module.__doc__ or ""
        assert "pgvector" in docstring.lower() or "PGVECTOR" in module.mem0_memory_service.__doc__

    def test_ingestion_instructions_present(self):
        """Verify ingestion instructions are present in module."""
        import bot.services.mem0_memory_service as module

        source = module.__file__
        with open(source) as f:
            content = f.read()

        assert "INGESTION INSTRUCTIONS" in content
        assert "WHAT TO STORE" in content
        assert "WHAT TO IGNORE" in content

    def test_method_docstrings_present(self):
        """Verify all public methods have docstrings."""
        from bot.services.mem0_memory_service import Mem0MemoryService

        assert Mem0MemoryService.write_factual.__doc__
        assert Mem0MemoryService.write_episodic.__doc__
        assert Mem0MemoryService.search.__doc__
