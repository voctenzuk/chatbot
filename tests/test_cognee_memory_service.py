"""Tests for Cognee Memory Service wrapper.

These tests use AsyncMock to avoid real network calls.
All Cognee API interactions are mocked.
"""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.services.cognee_memory_service import (
    CogneeMemoryService,
    get_memory_service,
    set_memory_service,
)
from bot.services.memory_models import MemoryCategory, MemoryType


class TestCogneeMemoryServiceUnit:
    """Unit tests for CogneeMemoryService using mocks."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock cognee client."""
        client = AsyncMock()
        client.add = AsyncMock(return_value=None)
        client.cognify = AsyncMock(return_value=None)
        client.search = AsyncMock(return_value=[])
        client.delete_dataset = AsyncMock(return_value=None)
        return client

    @pytest.fixture
    def service(self, mock_client):
        """Create service with mock client."""
        return CogneeMemoryService(client=mock_client)

    def test_init_with_mock_client(self, mock_client):
        """Test initialization with mock client."""
        service = CogneeMemoryService(client=mock_client)
        assert service._client is mock_client

    def test_cognee_not_available_raises(self):
        """Test that RuntimeError is raised when cognee not installed."""
        with patch("bot.services.cognee_memory_service.COGNEE_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="cognee package is not installed"):
                CogneeMemoryService()

    def test_user_dataset(self, service):
        """Test user dataset name formatting."""
        assert service._user_dataset(12345) == "tg_user_12345"
        assert service._user_dataset(0) == "tg_user_0"
        assert service._user_dataset(999999) == "tg_user_999999"

    @pytest.mark.asyncio
    async def test_write_factual_success(self, service, mock_client):
        """Test writing factual memory successfully."""
        memory_id = await service.write_factual(
            content="User likes pizza",
            user_id=123,
            metadata={"source": "conversation"},
            importance=1.5,
            tags=["food", "preference"],
        )

        assert memory_id  # Non-empty string
        mock_client.add.assert_called_once_with("User likes pizza", "tg_user_123")

    @pytest.mark.asyncio
    async def test_write_factual_default_values(self, service, mock_client):
        """Test writing factual memory with default values."""
        memory_id = await service.write_factual(
            content="User is a developer",
            user_id=456,
        )

        assert memory_id
        mock_client.add.assert_called_once_with("User is a developer", "tg_user_456")

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
    async def test_write_factual_tracks_pending(self, service, mock_client):
        """Test that write_factual marks dataset as pending cognify."""
        await service.write_factual(content="Test", user_id=123)

        assert "tg_user_123" in service._pending_datasets

    @pytest.mark.asyncio
    async def test_write_episodic_success(self, service, mock_client):
        """Test writing episodic memory successfully."""
        memory_id = await service.write_episodic(
            content="User asked about Python async patterns",
            user_id=123,
            run_id="session_abc",
            metadata={"topic": "programming"},
            importance=1.2,
            tags=["python", "async"],
            emotional_valence=0.8,
        )

        assert memory_id
        mock_client.add.assert_called_once_with(
            "User asked about Python async patterns",
            "tg_user_123",
        )

    @pytest.mark.asyncio
    async def test_write_episodic_default_values(self, service, mock_client):
        """Test writing episodic memory with default values."""
        memory_id = await service.write_episodic(
            content="Conversation event",
            user_id=789,
            run_id="run_xyz",
        )

        assert memory_id
        mock_client.add.assert_called_once_with("Conversation event", "tg_user_789")

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
    async def test_search_success_with_strings(self, service, mock_client):
        """Test searching when cognee returns plain strings."""
        mock_client.search.return_value = [
            "User likes pizza",
            "User enjoys hiking",
        ]

        results = await service.search(
            query="What does user like?",
            user_id=123,
            limit=5,
        )

        assert len(results) == 2
        assert results[0].content == "User likes pizza"
        assert results[0].user_id == 123
        assert results[0].memory_type == MemoryType.TEXT
        assert results[0].memory_category == MemoryCategory.SEMANTIC
        mock_client.search.assert_called_once_with(
            query_text="What does user like?", datasets=["tg_user_123"]
        )

    @pytest.mark.asyncio
    async def test_search_success_with_dicts(self, service, mock_client):
        """Test searching when cognee returns dict results."""
        mock_client.search.return_value = [
            {"text": "User likes pizza"},
            {"text": "User enjoys hiking"},
        ]

        results = await service.search(query="likes", user_id=123)

        assert len(results) == 2
        assert results[0].content == "User likes pizza"

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, service, mock_client):
        """Test that search respects the limit parameter."""
        mock_client.search.return_value = ["A", "B", "C", "D", "E"]

        results = await service.search(query="test", user_id=123, limit=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_empty_results(self, service, mock_client):
        """Test searching with no results."""
        mock_client.search.return_value = []

        results = await service.search(query="nonexistent", user_id=123)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self, service, mock_client):
        """Test that search errors return empty list instead of raising."""
        mock_client.search.side_effect = Exception("Search failed")

        results = await service.search(query="test", user_id=123)

        assert results == []

    @pytest.mark.asyncio
    async def test_cognify_success(self, service, mock_client):
        """Test cognify clears pending datasets."""
        service._pending_datasets.add("tg_user_123")
        service._pending_datasets.add("tg_user_456")

        await service.cognify()

        mock_client.cognify.assert_called_once()
        call_kwargs = mock_client.cognify.call_args.kwargs
        assert set(call_kwargs["datasets"]) == {"tg_user_123", "tg_user_456"}
        assert len(service._pending_datasets) == 0

    @pytest.mark.asyncio
    async def test_cognify_skips_when_no_pending(self, service, mock_client):
        """Test cognify does nothing when no pending datasets."""
        await service.cognify()

        mock_client.cognify.assert_not_called()

    @pytest.mark.asyncio
    async def test_cognify_concurrent_calls_safe(self, service, mock_client):
        """Test that concurrent cognify calls don't lose pending datasets."""
        service._pending_datasets.add("tg_user_1")
        service._pending_datasets.add("tg_user_2")

        await asyncio.gather(service.cognify(), service.cognify())

        assert len(service._pending_datasets) == 0

    @pytest.mark.asyncio
    async def test_cognify_error(self, service, mock_client):
        """Test cognify error propagation."""
        service._pending_datasets.add("tg_user_123")
        mock_client.cognify.side_effect = Exception("Cognify failed")

        with pytest.raises(Exception, match="Cognify failed"):
            await service.cognify()

    @pytest.mark.asyncio
    async def test_get_all_memories_success(self, service, mock_client):
        """Test getting all memories."""
        mock_client.search.return_value = ["Memory 1", "Memory 2"]

        results = await service.get_all_memories(user_id=123, limit=10)

        assert len(results) == 2
        assert results[0].content == "Memory 1"
        assert results[1].content == "Memory 2"

    @pytest.mark.asyncio
    async def test_get_all_memories_error_returns_empty(self, service, mock_client):
        """Test that get_all errors return empty list."""
        mock_client.search.side_effect = Exception("API Error")

        results = await service.get_all_memories(user_id=123)

        assert results == []

    @pytest.mark.asyncio
    async def test_delete_user_memories_success(self, service, mock_client):
        """Test deleting all user memories."""
        service._pending_datasets.add("tg_user_123")

        await service.delete_user_memories(user_id=123)

        mock_client.delete_dataset.assert_called_once_with("tg_user_123")
        assert "tg_user_123" not in service._pending_datasets

    @pytest.mark.asyncio
    async def test_delete_user_memories_error(self, service, mock_client):
        """Test error handling when deleting memories."""
        mock_client.delete_dataset.side_effect = Exception("Delete failed")

        with pytest.raises(Exception, match="Delete failed"):
            await service.delete_user_memories(user_id=123)


class TestCogneeMemoryServiceIntegrationPatterns:
    """Integration pattern tests demonstrating usage."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock cognee client."""
        client = AsyncMock()
        client.add = AsyncMock(return_value=None)
        client.cognify = AsyncMock(return_value=None)
        client.search = AsyncMock(return_value=[])
        client.delete_dataset = AsyncMock(return_value=None)
        return client

    @pytest.fixture
    def service(self, mock_client):
        """Create service with mock client."""
        return CogneeMemoryService(client=mock_client)

    @pytest.mark.asyncio
    async def test_fact_and_episodic_workflow(self, service, mock_client):
        """Test combined workflow of factual and episodic memories."""
        mock_client.search.return_value = [
            "User likes Python",
            "Discussed async",
        ]

        # Store user preference (factual)
        fact_id = await service.write_factual(
            content="User likes Python programming",
            user_id=123,
            tags=["programming", "python"],
            importance=1.5,
        )
        assert fact_id

        # Store conversation events (episodic)
        run_id = "session_2024_01_15"
        ep_id1 = await service.write_episodic(
            content="User and bot discussed Python async patterns",
            user_id=123,
            run_id=run_id,
            emotional_valence=0.6,
        )
        assert ep_id1

        ep_id2 = await service.write_episodic(
            content="User was excited about asyncio library",
            user_id=123,
            run_id=run_id,
            emotional_valence=0.9,
        )
        assert ep_id2

        # Search for relevant memories
        results = await service.search(
            query="What does user like about Python?",
            user_id=123,
        )
        assert len(results) == 2

        # All adds go to the same user dataset
        assert mock_client.add.call_count == 3
        for call in mock_client.add.call_args_list:
            assert call.args[1] == "tg_user_123"

    @pytest.mark.asyncio
    async def test_write_cognify_search_flow(self, service, mock_client):
        """Test full cognee pipeline: add → cognify → search."""
        mock_client.search.return_value = ["Cognified result"]

        await service.write_factual(content="Important fact", user_id=42)
        assert "tg_user_42" in service._pending_datasets

        await service.cognify()
        assert len(service._pending_datasets) == 0

        results = await service.search(query="fact", user_id=42)
        assert len(results) == 1
        assert results[0].content == "Cognified result"


class TestCogneeMemoryServiceGlobalInstance:
    """Tests for global service instance management."""

    def setup_method(self):
        """Reset global instance before each test."""
        set_memory_service(None)

    def teardown_method(self):
        """Reset global instance after each test."""
        set_memory_service(None)

    def test_set_memory_service(self):
        """Test setting global instance."""
        mock_client = AsyncMock()
        custom_service = CogneeMemoryService(client=mock_client)

        set_memory_service(custom_service)

        retrieved = get_memory_service()
        assert retrieved is custom_service

    def test_set_memory_service_to_none_resets(self):
        """Test resetting global instance to None."""
        mock_client = AsyncMock()
        custom_service = CogneeMemoryService(client=mock_client)

        set_memory_service(custom_service)
        set_memory_service(None)

        with patch("bot.services.cognee_memory_service.CogneeMemoryService") as mock_cls:
            mock_cls.return_value = AsyncMock()
            get_memory_service()
            mock_cls.assert_called_once()


class TestExtractText:
    """Tests for _extract_text static method."""

    def test_extract_from_string(self):
        """Test extracting text from plain string."""
        assert CogneeMemoryService._extract_text("hello") == "hello"

    def test_extract_from_dict_text_key(self):
        """Test extracting text from dict with 'text' key."""
        assert CogneeMemoryService._extract_text({"text": "hello"}) == "hello"

    def test_extract_from_dict_content_key(self):
        """Test extracting text from dict with 'content' key."""
        assert CogneeMemoryService._extract_text({"content": "hello"}) == "hello"

    def test_extract_from_dict_memory_key(self):
        """Test extracting text from dict with 'memory' key."""
        assert CogneeMemoryService._extract_text({"memory": "hello"}) == "hello"

    def test_extract_from_object_with_search_result(self):
        """Test extracting text from object with search_result attr."""
        obj = MagicMock()
        obj.search_result = "result text"
        assert CogneeMemoryService._extract_text(obj) == "result text"

    def test_extract_from_object_with_search_result_list(self):
        """Test extracting text from object with search_result list."""
        obj = MagicMock()
        obj.search_result = [{"text": "chunk text"}]
        assert CogneeMemoryService._extract_text(obj) == "chunk text"

    def test_extract_fallback_to_str(self):
        """Test fallback to str() for unknown types."""
        result = CogneeMemoryService._extract_text(42)
        assert result == "42"
