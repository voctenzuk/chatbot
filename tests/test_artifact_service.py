"""Tests for ArtifactService and storage backend."""

import io
import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from bot.services.artifact_service import (
    Artifact,
    ArtifactProcessingStatus,
    ArtifactService,
    ArtifactText,
    ArtifactType,
    CreateArtifactRequest,
    TextSurrogateForContext,
    TextSurrogateKind,
    get_artifact_service,
    set_artifact_service,
)
from bot.services.storage_backend import (
    LocalStorageBackend,
    StorageReference,
    get_storage_backend,
    set_storage_backend,
)


class TestArtifact:
    """Tests for Artifact dataclass."""

    def test_create_artifact(self):
        """Test creating an Artifact."""
        artifact = Artifact(
            id="test-uuid",
            user_id=12345,
            type=ArtifactType.IMAGE,
            mime_type="image/jpeg",
            size_bytes=1024,
            sha256="abc123",
            storage_key="12345/test-uuid/photo.jpg",
            storage_provider="local",
        )

        assert artifact.id == "test-uuid"
        assert artifact.user_id == 12345
        assert artifact.type == ArtifactType.IMAGE
        assert artifact.mime_type == "image/jpeg"
        assert artifact.size_bytes == 1024
        assert artifact.sha256 == "abc123"
        assert artifact.processing_status == ArtifactProcessingStatus.PENDING

    def test_artifact_from_row(self):
        """Test creating Artifact from database row."""
        row = {
            "id": "test-uuid",
            "user_id": 12345,
            "type": "image",
            "mime_type": "image/jpeg",
            "size_bytes": 1024,
            "sha256": "abc123",
            "storage_key": "12345/test-uuid/photo.jpg",
            "storage_provider": "local",
            "processing_status": "completed",
            "original_filename": "photo.jpg",
            "created_at": "2024-01-01T00:00:00Z",
        }

        artifact = Artifact.from_row(row)

        assert artifact.id == "test-uuid"
        assert artifact.type == ArtifactType.IMAGE
        assert artifact.processing_status == ArtifactProcessingStatus.COMPLETED
        assert artifact.original_filename == "photo.jpg"


class TestArtifactText:
    """Tests for ArtifactText dataclass."""

    def test_create_artifact_text(self):
        """Test creating an ArtifactText."""
        text = ArtifactText(
            id="text-uuid",
            artifact_id="artifact-uuid",
            text_kind=TextSurrogateKind.VISION_SUMMARY,
            text_content="A photo of a cat",
        )

        assert text.id == "text-uuid"
        assert text.artifact_id == "artifact-uuid"
        assert text.text_kind == TextSurrogateKind.VISION_SUMMARY
        assert text.text_content == "A photo of a cat"
        assert not text.is_chunk
        assert text.chunk_info is None

    def test_chunk_properties(self):
        """Test chunk-related properties."""
        text = ArtifactText(
            id="text-uuid",
            artifact_id="artifact-uuid",
            text_kind=TextSurrogateKind.TEXT_CHUNK,
            text_content="Chunk content",
            chunk_index=2,
            chunk_total=5,
        )

        assert text.is_chunk
        assert text.chunk_info == "chunk 3/5"

    def test_artifact_text_from_row(self):
        """Test creating ArtifactText from database row."""
        row = {
            "id": "text-uuid",
            "artifact_id": "artifact-uuid",
            "text_kind": "vision_summary",
            "text_content": "A photo of a cat",
            "confidence": 0.95,
            "model_used": "gpt-4-vision",
            "created_at": "2024-01-01T00:00:00Z",
        }

        text = ArtifactText.from_row(row)

        assert text.text_kind == TextSurrogateKind.VISION_SUMMARY
        assert text.confidence == 0.95
        assert text.model_used == "gpt-4-vision"


class TestTextSurrogateForContext:
    """Tests for TextSurrogateForContext dataclass."""

    def test_create_surrogate(self):
        """Test creating a context surrogate."""
        surrogate = TextSurrogateForContext(
            artifact_id="artifact-uuid",
            artifact_type=ArtifactType.IMAGE,
            original_filename="photo.jpg",
            text_kind=TextSurrogateKind.VISION_SUMMARY,
            text_content="A photo of a cat",
        )

        assert surrogate.artifact_id == "artifact-uuid"
        assert surrogate.artifact_type == ArtifactType.IMAGE

    def test_to_context_string(self):
        """Test converting to context string."""
        surrogate = TextSurrogateForContext(
            artifact_id="artifact-uuid",
            artifact_type=ArtifactType.IMAGE,
            original_filename="photo.jpg",
            text_kind=TextSurrogateKind.VISION_SUMMARY,
            text_content="A photo of a cat",
        )

        context = surrogate.to_context_string()

        assert "[image: photo.jpg]" in context
        assert "A photo of a cat" in context

    def test_to_context_string_with_chunk(self):
        """Test context string with chunk info."""
        surrogate = TextSurrogateForContext(
            artifact_id="artifact-uuid",
            artifact_type=ArtifactType.DOCUMENT,
            original_filename="doc.pdf",
            text_kind=TextSurrogateKind.TEXT_CHUNK,
            text_content="Document content",
            chunk_info="chunk 1/3",
        )

        context = surrogate.to_context_string()

        assert "[document: doc.pdf]" in context
        assert "(chunk 1/3)" in context
        assert "Document content" in context


class TestCreateArtifactRequest:
    """Tests for CreateArtifactRequest dataclass."""

    def test_create_request(self):
        """Test creating a request."""
        data = b"test image data"
        request = CreateArtifactRequest(
            user_id=12345,
            type=ArtifactType.IMAGE,
            mime_type="image/jpeg",
            filename="photo.jpg",
            data=data,
            episode_id="episode-uuid",
        )

        assert request.user_id == 12345
        assert request.type == ArtifactType.IMAGE
        assert request.data == data
        assert request.episode_id == "episode-uuid"

    def test_create_request_with_file_object(self):
        """Test creating a request with file-like object."""
        data = io.BytesIO(b"test image data")
        request = CreateArtifactRequest(
            user_id=12345,
            type=ArtifactType.DOCUMENT,
            mime_type="application/pdf",
            filename="doc.pdf",
            data=data,
        )

        assert request.type == ArtifactType.DOCUMENT


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create a temporary storage backend."""
        backend = LocalStorageBackend(base_path=tmp_path / "artifacts")
        return backend

    @pytest.mark.asyncio
    async def test_store_bytes(self, temp_storage):
        """Test storing bytes data."""
        data = b"test file content"
        ref = await temp_storage.store(
            data=data,
            user_id=12345,
            artifact_id="test-artifact",
            filename="test.txt",
        )

        assert ref.storage_key == "12345/test-artifact/test.txt"
        assert ref.provider == "local"
        assert ref.size_bytes == len(data)
        assert len(ref.sha256) == 64  # SHA256 hex length

    @pytest.mark.asyncio
    async def test_store_file_object(self, temp_storage):
        """Test storing file-like object."""
        data = io.BytesIO(b"test file content")
        ref = await temp_storage.store(
            data=data,
            user_id=12345,
            artifact_id="test-artifact",
            filename="test.txt",
        )

        assert ref.size_bytes == len(b"test file content")

    @pytest.mark.asyncio
    async def test_retrieve_existing(self, temp_storage):
        """Test retrieving existing file."""
        data = b"test content"
        ref = await temp_storage.store(
            data=data,
            user_id=12345,
            artifact_id="test-artifact",
            filename="test.txt",
        )

        retrieved = await temp_storage.retrieve(ref.storage_key)

        assert retrieved == data

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, temp_storage):
        """Test retrieving nonexistent file."""
        retrieved = await temp_storage.retrieve("nonexistent/key.txt")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_exists(self, temp_storage):
        """Test checking file existence."""
        data = b"test content"
        ref = await temp_storage.store(
            data=data,
            user_id=12345,
            artifact_id="test-artifact",
            filename="test.txt",
        )

        assert await temp_storage.exists(ref.storage_key) is True
        assert await temp_storage.exists("nonexistent/key.txt") is False

    @pytest.mark.asyncio
    async def test_delete(self, temp_storage):
        """Test deleting file."""
        data = b"test content"
        ref = await temp_storage.store(
            data=data,
            user_id=12345,
            artifact_id="test-artifact",
            filename="test.txt",
        )

        deleted = await temp_storage.delete(ref.storage_key)

        assert deleted is True
        assert await temp_storage.exists(ref.storage_key) is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, temp_storage):
        """Test deleting nonexistent file."""
        deleted = await temp_storage.delete("nonexistent/key.txt")

        assert deleted is False

    def test_get_public_url(self, temp_storage):
        """Test that public URL is None for local storage."""
        url = temp_storage.get_public_url("12345/test/file.txt")

        assert url is None

    def test_compute_sha256_bytes(self, temp_storage):
        """Test computing SHA256 from bytes."""
        data = b"test content"
        sha256 = temp_storage.compute_sha256(data)

        assert len(sha256) == 64
        # SHA256 is deterministic
        assert temp_storage.compute_sha256(data) == sha256

    def test_compute_sha256_file_object(self, temp_storage):
        """Test computing SHA256 from file object."""
        data = io.BytesIO(b"test content")
        sha256 = temp_storage.compute_sha256(data)

        assert len(sha256) == 64
        # File pointer should be reset
        assert data.read() == b"test content"

    def test_path_traversal_protection(self, temp_storage):
        """Test protection against path traversal."""
        with pytest.raises(ValueError, match="path traversal"):
            temp_storage._get_file_path("../../../etc/passwd")


class TestArtifactServiceBasics:
    """Basic tests for ArtifactService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database client."""
        db = MagicMock()
        db._client = MagicMock()
        return db

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        storage = AsyncMock()
        storage.compute_sha256.return_value = "abc123" * 8  # 64 chars
        return storage

    @pytest.fixture
    def service(self, mock_db, mock_storage):
        """Create an ArtifactService with mocked dependencies."""
        return ArtifactService(db_client=mock_db, storage_backend=mock_storage)

    @pytest.mark.asyncio
    async def test_get_artifact_by_id_found(self, service, mock_db):
        """Test retrieving artifact by ID when found."""
        mock_db._client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = {
            "id": "test-uuid",
            "user_id": 12345,
            "type": "image",
            "mime_type": "image/jpeg",
            "size_bytes": 1024,
            "sha256": "abc123",
            "storage_key": "key",
            "storage_provider": "local",
        }

        artifact = await service.get_artifact_by_id("test-uuid")

        assert artifact is not None
        assert artifact.id == "test-uuid"
        assert artifact.type == ArtifactType.IMAGE

    @pytest.mark.asyncio
    async def test_get_artifact_by_id_not_found(self, service, mock_db):
        """Test retrieving artifact by ID when not found."""
        mock_db._client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = None

        artifact = await service.get_artifact_by_id("nonexistent")

        assert artifact is None


class TestArtifactServiceDeduplication:
    """Tests for artifact deduplication."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database client."""
        db = MagicMock()
        db._client = MagicMock()
        return db

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        storage = AsyncMock()
        storage.compute_sha256.return_value = "abc123" * 8
        return storage

    @pytest.fixture
    def service(self, mock_db, mock_storage):
        """Create an ArtifactService with mocked dependencies."""
        return ArtifactService(db_client=mock_db, storage_backend=mock_storage)

    @pytest.mark.asyncio
    async def test_deduplication_reuses_existing(self, service, mock_db, mock_storage):
        """Test that duplicate sha256 returns existing artifact."""
        # Setup existing artifact in DB
        existing_row = {
            "id": "existing-uuid",
            "user_id": 12345,
            "type": "image",
            "mime_type": "image/jpeg",
            "size_bytes": 1024,
            "sha256": "abc123" * 8,
            "storage_key": "existing-key",
            "storage_provider": "local",
        }

        # Mock get_artifact_by_sha256 RPC
        mock_db._client.rpc.return_value.execute.return_value.data = [
            {"artifact_id": "existing-uuid", "storage_key": "existing-key", "created_at": "2024-01-01T00:00:00Z"}
        ]

        # Mock get_artifact_by_id
        mock_db._client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = existing_row

        request = CreateArtifactRequest(
            user_id=12345,
            type=ArtifactType.IMAGE,
            mime_type="image/jpeg",
            filename="photo.jpg",
            data=b"same content",
        )

        result = await service.create_artifact(request)

        assert result.is_duplicate is True
        assert result.artifact.id == "existing-uuid"
        # Should not call storage.store for duplicates
        mock_storage.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_artifact_stores_file(self, service, mock_db, mock_storage):
        """Test that new artifacts store the file."""
        # No existing artifact
        mock_db._client.rpc.return_value.execute.return_value.data = []

        # Mock storage.store
        mock_storage.store.return_value = StorageReference(
            storage_key="12345/new-uuid/photo.jpg",
            provider="local",
            size_bytes=1024,
            sha256="abc123" * 8,
        )

        # Mock add_artifact RPC
        mock_db._client.rpc.return_value.execute.return_value.data = "new-artifact-uuid"

        # Mock get_artifact_by_id for the return
        mock_db._client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = {
            "id": "new-artifact-uuid",
            "user_id": 12345,
            "type": "image",
            "mime_type": "image/jpeg",
            "size_bytes": 1024,
            "sha256": "abc123" * 8,
            "storage_key": "12345/new-uuid/photo.jpg",
            "storage_provider": "local",
        }

        request = CreateArtifactRequest(
            user_id=12345,
            type=ArtifactType.IMAGE,
            mime_type="image/jpeg",
            filename="photo.jpg",
            data=b"new content",
        )

        result = await service.create_artifact(request)

        assert result.is_duplicate is False
        mock_storage.store.assert_called_once()


class TestArtifactServiceTextSurrogates:
    """Tests for text surrogate operations."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database client."""
        db = MagicMock()
        db._client = MagicMock()
        return db

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_db, mock_storage):
        """Create an ArtifactService with mocked dependencies."""
        return ArtifactService(db_client=mock_db, storage_backend=mock_storage)

    @pytest.mark.asyncio
    async def test_add_vision_summary(self, service, mock_db):
        """Test adding vision summary."""
        mock_db._client.rpc.return_value.execute.return_value.data = "text-uuid"

        # Mock _get_text_by_id
        service._get_text_by_id = AsyncMock(return_value=ArtifactText(
            id="text-uuid",
            artifact_id="artifact-uuid",
            text_kind=TextSurrogateKind.VISION_SUMMARY,
            text_content="A photo of a cat",
        ))

        text = await service.add_vision_summary(
            artifact_id="artifact-uuid",
            summary="A photo of a cat",
            is_short=True,
            model_used="gpt-4-vision",
        )

        assert text.text_kind == TextSurrogateKind.VISION_SUMMARY
        assert text.text_content == "A photo of a cat"

    @pytest.mark.asyncio
    async def test_add_vision_detail(self, service, mock_db):
        """Test adding detailed vision description."""
        mock_db._client.rpc.return_value.execute.return_value.data = "text-uuid"

        service._get_text_by_id = AsyncMock(return_value=ArtifactText(
            id="text-uuid",
            artifact_id="artifact-uuid",
            text_kind=TextSurrogateKind.VISION_DETAIL,
            text_content="Detailed description",
        ))

        text = await service.add_vision_summary(
            artifact_id="artifact-uuid",
            summary="Detailed description",
            is_short=False,
        )

        assert text.text_kind == TextSurrogateKind.VISION_DETAIL

    @pytest.mark.asyncio
    async def test_add_ocr_text(self, service, mock_db):
        """Test adding OCR text."""
        mock_db._client.rpc.return_value.execute.return_value.data = "text-uuid"

        service._get_text_by_id = AsyncMock(return_value=ArtifactText(
            id="text-uuid",
            artifact_id="artifact-uuid",
            text_kind=TextSurrogateKind.OCR_TEXT,
            text_content="Extracted text",
            confidence=0.95,
            model_used="tesseract",
        ))

        text = await service.add_ocr_text(
            artifact_id="artifact-uuid",
            ocr_text="Extracted text",
            confidence=0.95,
            model_used="tesseract",
        )

        assert text.text_kind == TextSurrogateKind.OCR_TEXT
        assert text.confidence == 0.95
        assert text.model_used == "tesseract"

    @pytest.mark.asyncio
    async def test_add_document_text(self, service, mock_db):
        """Test adding document text with summary."""
        mock_db._client.rpc.return_value.execute.return_value.data = "text-uuid"

        service._get_text_by_id = AsyncMock(side_effect=[
            ArtifactText(
                id="text-uuid-1",
                artifact_id="artifact-uuid",
                text_kind=TextSurrogateKind.EXTRACTED_TEXT,
                text_content="Full document text",
            ),
            ArtifactText(
                id="text-uuid-2",
                artifact_id="artifact-uuid",
                text_kind=TextSurrogateKind.FILE_SUMMARY,
                text_content="Document summary",
            ),
        ])

        texts = await service.add_document_text(
            artifact_id="artifact-uuid",
            extracted_text="Full document text",
            summary="Document summary",
        )

        assert len(texts) == 2
        assert texts[0].text_kind == TextSurrogateKind.EXTRACTED_TEXT
        assert texts[1].text_kind == TextSurrogateKind.FILE_SUMMARY

    @pytest.mark.asyncio
    async def test_add_document_chunk(self, service, mock_db):
        """Test adding chunked document text."""
        mock_db._client.rpc.return_value.execute.return_value.data = "text-uuid"

        service._get_text_by_id = AsyncMock(return_value=ArtifactText(
            id="text-uuid",
            artifact_id="artifact-uuid",
            text_kind=TextSurrogateKind.TEXT_CHUNK,
            text_content="Chunk content",
            chunk_index=0,
            chunk_total=3,
        ))

        texts = await service.add_document_text(
            artifact_id="artifact-uuid",
            extracted_text="Chunk content",
            chunk_index=0,
            chunk_total=3,
        )

        assert len(texts) == 1
        assert texts[0].text_kind == TextSurrogateKind.TEXT_CHUNK
        assert texts[0].chunk_index == 0
        assert texts[0].chunk_total == 3


class TestArtifactServiceContextIntegration:
    """Tests for context building integration."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database client."""
        db = MagicMock()
        db._client = MagicMock()
        return db

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_db, mock_storage):
        """Create an ArtifactService with mocked dependencies."""
        return ArtifactService(db_client=mock_db, storage_backend=mock_storage)

    @pytest.mark.asyncio
    async def test_get_surrogates_for_context(self, service, mock_db):
        """Test getting surrogates formatted for context."""
        mock_db._client.rpc.return_value.execute.return_value.data = [
            {
                "artifact_id": "artifact-1",
                "artifact_type": "image",
                "original_filename": "photo.jpg",
                "text_kind": "vision_summary",
                "text_content": "A photo of a cat",
                "chunk_info": None,
            },
            {
                "artifact_id": "artifact-2",
                "artifact_type": "document",
                "original_filename": "doc.pdf",
                "text_kind": "file_summary",
                "text_content": "Document about cats",
                "chunk_info": None,
            },
        ]

        surrogates = await service.get_surrogates_for_context(
            episode_id="episode-uuid",
            max_per_artifact=2,
            max_total=5,
        )

        assert len(surrogates) == 2
        assert surrogates[0].artifact_type == ArtifactType.IMAGE
        assert surrogates[0].text_content == "A photo of a cat"

    @pytest.mark.asyncio
    async def test_get_surrogates_empty(self, service, mock_db):
        """Test getting surrogates when none exist."""
        mock_db._client.rpc.return_value.execute.return_value.data = []

        surrogates = await service.get_surrogates_for_context(
            episode_id="episode-uuid",
        )

        assert surrogates == []


class TestGlobalInstance:
    """Tests for global ArtifactService instance management."""

    def test_get_artifact_service_creates_instance(self, monkeypatch):
        """Test that get_artifact_service creates default instance."""
        set_artifact_service(None)  # Reset

        # Mock dependencies to avoid actual initialization
        mock_db = MagicMock()
        mock_storage = MagicMock()

        # Patch the get functions
        monkeypatch.setattr("bot.services.artifact_service.get_db_client", lambda: mock_db)
        monkeypatch.setattr("bot.services.artifact_service.get_storage_backend", lambda: mock_storage)

        service = get_artifact_service()

        assert isinstance(service, ArtifactService)

    def test_set_artifact_service(self):
        """Test setting global instance."""
        mock_service = MagicMock(spec=ArtifactService)
        set_artifact_service(mock_service)

        retrieved = get_artifact_service()
        assert retrieved is mock_service

    def teardown_method(self):
        """Clean up after each test."""
        set_artifact_service(None)
