"""ArtifactService for managing file uploads and text surrogates.

This module provides the ArtifactService class for:
- Creating and retrieving artifacts with sha256 deduplication
- Managing text surrogates (vision summaries, OCR, document extraction)
- Integration with storage backends (local, S3)
- Context building support for artifact surrogates
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, BinaryIO

from loguru import logger

from bot.services.db_client import DatabaseClient, get_db_client
from bot.services.storage_backend import StorageBackend, StorageReference, get_storage_backend


class ArtifactType(Enum):
    """Type of artifact file."""

    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"


class ArtifactProcessingStatus(Enum):
    """Processing status of an artifact."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TextSurrogateKind(Enum):
    """Kind of text surrogate/extracted content."""

    # User-provided
    USER_CAPTION = "user_caption"

    # Image analysis
    VISION_SUMMARY = "vision_summary"  # Short description (1-2 lines)
    VISION_DETAIL = "vision_detail"  # Detailed description (paragraph)
    OCR_TEXT = "ocr_text"  # OCR extracted text

    # Document analysis
    FILE_SUMMARY = "file_summary"  # Document summary
    EXTRACTED_TEXT = "extracted_text"  # Full extracted text
    TEXT_CHUNK = "text_chunk"  # Chunk of extracted text

    # Audio/Video
    TRANSCRIPT = "transcript"

    # Auto-generated
    AUTO_CAPTION = "auto_caption"


@dataclass
class Artifact:
    """Artifact model representing a stored file."""

    id: str
    user_id: int
    type: ArtifactType
    mime_type: str
    size_bytes: int
    sha256: str
    storage_key: str
    storage_provider: str

    # Context references (nullable for orphaned/reused artifacts)
    thread_id: str | None = None
    episode_id: str | None = None
    message_id: str | None = None

    # Metadata
    original_filename: str | None = None
    processing_status: ArtifactProcessingStatus = ArtifactProcessingStatus.PENDING
    processing_error: str | None = None

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Artifact:
        """Create Artifact from database row."""
        return cls(
            id=str(row["id"]),
            user_id=row["user_id"],
            type=ArtifactType(row["type"]),
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            storage_key=row["storage_key"],
            storage_provider=row["storage_provider"],
            thread_id=str(row["thread_id"]) if row.get("thread_id") else None,
            episode_id=str(row["episode_id"]) if row.get("episode_id") else None,
            message_id=str(row["message_id"]) if row.get("message_id") else None,
            original_filename=row.get("original_filename"),
            processing_status=ArtifactProcessingStatus(row.get("processing_status", "pending")),
            processing_error=row.get("processing_error"),
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            if row.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
            if row.get("updated_at")
            else None,
        )


@dataclass
class ArtifactText:
    """Text surrogate or extracted content from an artifact."""

    id: str
    artifact_id: str
    text_kind: TextSurrogateKind
    text_content: str

    # Chunking support
    chunk_index: int | None = None
    chunk_total: int | None = None

    # Optional embedding (for semantic search)
    embedding: list[float] | None = None

    # Metadata
    confidence: float | None = None
    model_used: str | None = None

    created_at: datetime | None = None

    @property
    def is_chunk(self) -> bool:
        """Check if this is a chunk of a larger document."""
        return self.chunk_index is not None

    @property
    def chunk_info(self) -> str | None:
        """Get human-readable chunk info (e.g., 'chunk 1/5')."""
        if self.chunk_total is not None and self.chunk_index is not None:
            return f"chunk {self.chunk_index + 1}/{self.chunk_total}"
        return None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ArtifactText:
        """Create ArtifactText from database row."""
        return cls(
            id=str(row["id"]),
            artifact_id=str(row["artifact_id"]),
            text_kind=TextSurrogateKind(row["text_kind"]),
            text_content=row["text_content"],
            chunk_index=row.get("chunk_index"),
            chunk_total=row.get("chunk_total"),
            embedding=row.get("embedding"),
            confidence=row.get("confidence"),
            model_used=row.get("model_used"),
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            if row.get("created_at")
            else None,
        )


@dataclass
class TextSurrogateForContext:
    """Simplified text surrogate for context building."""

    artifact_id: str
    artifact_type: ArtifactType
    original_filename: str | None
    text_kind: TextSurrogateKind
    text_content: str
    chunk_info: str | None = None

    def to_context_string(self) -> str:
        """Convert to string suitable for LLM context.

        Returns:
            Formatted context string.
        """
        parts = []

        # Artifact reference
        if self.original_filename:
            parts.append(f"[{self.artifact_type.value}: {self.original_filename}]")
        else:
            parts.append(f"[{self.artifact_type.value}]")

        # Content with chunk info
        if self.chunk_info:
            parts.append(f"({self.chunk_info})")
        parts.append(self.text_content)

        return " ".join(parts)


@dataclass
class CreateArtifactRequest:
    """Request to create a new artifact."""

    user_id: int
    type: ArtifactType
    mime_type: str
    filename: str
    data: bytes | BinaryIO

    # Optional context references
    thread_id: str | None = None
    episode_id: str | None = None
    message_id: str | None = None


@dataclass
class CreateArtifactResult:
    """Result of artifact creation."""

    artifact: Artifact
    is_duplicate: bool  # True if this was a deduplication (existing artifact reused)
    storage_ref: StorageReference


class ArtifactService:
    """Service for managing artifacts and text surrogates.

    Features:
    - Create artifacts with automatic sha256 deduplication
    - Store files via pluggable storage backend (local, S3)
    - Manage text surrogates (vision summaries, OCR, document extraction)
    - Retrieve artifacts for context building
    """

    def __init__(
        self,
        db_client: DatabaseClient | None = None,
        storage_backend: StorageBackend | None = None,
    ) -> None:
        """Initialize ArtifactService.

        Args:
            db_client: Database client. Uses global if not provided.
            storage_backend: Storage backend. Uses global if not provided.
        """
        self._db = db_client or get_db_client()
        self._storage = storage_backend or get_storage_backend()
        logger.info("ArtifactService initialized")

    # -------------------------------------------------------------------------
    # Artifact CRUD
    # -------------------------------------------------------------------------

    async def create_artifact(
        self,
        request: CreateArtifactRequest,
    ) -> CreateArtifactResult:
        """Create a new artifact with deduplication.

        If an artifact with the same sha256 hash exists for this user,
        the existing artifact is returned (deduplication).

        Args:
            request: Artifact creation request.

        Returns:
            CreateArtifactResult with artifact and deduplication info.
        """
        # Compute hash for deduplication check
        sha256 = self._storage.compute_sha256(request.data)

        # Check for existing artifact with same hash
        existing = await self.get_artifact_by_sha256(request.user_id, sha256)
        if existing:
            logger.info(
                "Artifact deduplication: reusing existing artifact {} for user {}",
                existing.id,
                request.user_id,
            )
            # Update context references if provided
            if request.thread_id or request.episode_id or request.message_id:
                existing = await self._update_artifact_context(
                    existing.id,
                    thread_id=request.thread_id,
                    episode_id=request.episode_id,
                    message_id=request.message_id,
                )
            return CreateArtifactResult(
                artifact=existing,
                is_duplicate=True,
                storage_ref=StorageReference(
                    storage_key=existing.storage_key,
                    provider=existing.storage_provider,
                    size_bytes=existing.size_bytes,
                    sha256=existing.sha256,
                ),
            )

        # Store the file
        # Generate a temporary artifact ID for storage path
        import uuid

        temp_artifact_id = str(uuid.uuid4())
        storage_ref = await self._storage.store(
            data=request.data,
            user_id=request.user_id,
            artifact_id=temp_artifact_id,
            filename=request.filename,
        )

        # Create artifact record in database
        try:
            artifact_id = await self._add_artifact_to_db(
                user_id=request.user_id,
                artifact_type=request.type,
                mime_type=request.mime_type,
                size_bytes=storage_ref.size_bytes,
                sha256=storage_ref.sha256,
                storage_key=storage_ref.storage_key,
                storage_provider=storage_ref.provider,
                original_filename=request.filename,
                thread_id=request.thread_id,
                episode_id=request.episode_id,
                message_id=request.message_id,
            )
        except Exception:
            # Rollback: delete stored file
            await self._storage.delete(storage_ref.storage_key)
            raise

        # Retrieve full artifact
        artifact = await self.get_artifact_by_id(artifact_id)
        if not artifact:
            raise RuntimeError(f"Failed to retrieve created artifact {artifact_id}")

        logger.info(
            "Created artifact {} for user {} (type={}, size={} bytes)",
            artifact_id,
            request.user_id,
            request.type.value,
            storage_ref.size_bytes,
        )

        return CreateArtifactResult(
            artifact=artifact,
            is_duplicate=False,
            storage_ref=storage_ref,
        )

    async def get_artifact_by_id(self, artifact_id: str) -> Artifact | None:
        """Get artifact by ID.

        Args:
            artifact_id: Artifact UUID.

        Returns:
            Artifact or None if not found.
        """
        try:
            response = (
                self._db._client.table("artifacts")
                .select("*")
                .eq("id", artifact_id)
                .maybe_single()
                .execute()
            )
            if response.data:
                return Artifact.from_row(response.data)
            return None
        except Exception as e:
            logger.error("Failed to get artifact {}: {}", artifact_id, e)
            return None

    async def get_artifact_by_sha256(self, user_id: int, sha256: str) -> Artifact | None:
        """Get artifact by sha256 hash (for deduplication).

        Args:
            user_id: Telegram user ID.
            sha256: SHA256 hash.

        Returns:
            Artifact or None if not found.
        """
        try:
            response = self._db._client.rpc(
                "get_artifact_by_sha256",
                {"p_user_id": user_id, "p_sha256": sha256},
            ).execute()

            if response.data and len(response.data) > 0:
                artifact_id = response.data[0]["artifact_id"]
                return await self.get_artifact_by_id(artifact_id)
            return None
        except Exception as e:
            logger.error("Failed to get artifact by sha256: {}", e)
            return None

    async def get_artifacts_for_episode(
        self,
        episode_id: str,
        text_kinds: list[TextSurrogateKind] | None = None,
    ) -> list[tuple[Artifact, list[ArtifactText]]]:
        """Get all artifacts for an episode with their text surrogates.

        Args:
            episode_id: Episode UUID.
            text_kinds: Optional filter for text kinds.

        Returns:
            List of (Artifact, list[ArtifactText]) tuples.
        """
        try:
            kind_strings = [k.value for k in text_kinds] if text_kinds else None
            response = self._db._client.rpc(
                "get_artifacts_for_episode",
                {
                    "p_episode_id": episode_id,
                    "p_text_kinds": kind_strings,
                },
            ).execute()

            results = []
            for row in response.data or []:
                artifact = Artifact.from_row(row)

                # Parse text surrogates JSONB
                text_surrogates = []
                surrogates_data = row.get("text_surrogates", [])
                if surrogates_data and surrogates_data != [{}]:
                    for surrogate_data in surrogates_data:
                        if surrogate_data and "text_kind" in surrogate_data:
                            text_surrogates.append(ArtifactText.from_row(surrogate_data))

                results.append((artifact, text_surrogates))

            return results
        except Exception as e:
            logger.error("Failed to get artifacts for episode {}: {}", episode_id, e)
            return []

    async def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact and its stored file.

        Args:
            artifact_id: Artifact UUID.

        Returns:
            True if deleted, False if not found.
        """
        artifact = await self.get_artifact_by_id(artifact_id)
        if not artifact:
            return False

        # Delete from storage
        await self._storage.delete(artifact.storage_key)

        # Delete from database (cascades to artifact_text)
        try:
            self._db._client.table("artifacts").delete().eq("id", artifact_id).execute()
            logger.info("Deleted artifact {}", artifact_id)
            return True
        except Exception as e:
            logger.error("Failed to delete artifact {}: {}", artifact_id, e)
            return False

    async def _add_artifact_to_db(
        self,
        user_id: int,
        artifact_type: ArtifactType,
        mime_type: str,
        size_bytes: int,
        sha256: str,
        storage_key: str,
        storage_provider: str,
        original_filename: str | None = None,
        thread_id: str | None = None,
        episode_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        """Add artifact record to database.

        Returns:
            Artifact ID.
        """
        response = self._db._client.rpc(
            "add_artifact",
            {
                "p_user_id": user_id,
                "p_type": artifact_type.value,
                "p_mime_type": mime_type,
                "p_size_bytes": size_bytes,
                "p_sha256": sha256,
                "p_storage_key": storage_key,
                "p_storage_provider": storage_provider,
                "p_original_filename": original_filename,
                "p_thread_id": thread_id,
                "p_episode_id": episode_id,
                "p_message_id": message_id,
            },
        ).execute()

        return response.data

    async def _update_artifact_context(
        self,
        artifact_id: str,
        thread_id: str | None = None,
        episode_id: str | None = None,
        message_id: str | None = None,
    ) -> Artifact:
        """Update artifact context references.

        Args:
            artifact_id: Artifact UUID.
            thread_id: Optional new thread ID.
            episode_id: Optional new episode ID.
            message_id: Optional new message ID.

        Returns:
            Updated Artifact.
        """
        updates: dict[str, Any] = {"updated_at": datetime.now().isoformat()}
        if thread_id:
            updates["thread_id"] = thread_id
        if episode_id:
            updates["episode_id"] = episode_id
        if message_id:
            updates["message_id"] = message_id

        response = (
            self._db._client.table("artifacts").update(updates).eq("id", artifact_id).execute()
        )

        return Artifact.from_row(response.data[0])

    # -------------------------------------------------------------------------
    # Text Surrogate Management
    # -------------------------------------------------------------------------

    async def add_text_surrogate(
        self,
        artifact_id: str,
        text_kind: TextSurrogateKind,
        text_content: str,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        embedding: list[float] | None = None,
        confidence: float | None = None,
        model_used: str | None = None,
    ) -> ArtifactText:
        """Add a text surrogate to an artifact.

        Args:
            artifact_id: Artifact UUID.
            text_kind: Type of text surrogate.
            text_content: The text content.
            chunk_index: Optional chunk index for chunked documents.
            chunk_total: Optional total chunks count.
            embedding: Optional embedding vector for semantic search.
            confidence: Optional confidence score (0.0-1.0).
            model_used: Optional model name used for extraction.

        Returns:
            Created ArtifactText.
        """
        response = self._db._client.rpc(
            "upsert_artifact_text",
            {
                "p_artifact_id": artifact_id,
                "p_text_kind": text_kind.value,
                "p_text_content": text_content,
                "p_chunk_index": chunk_index,
                "p_chunk_total": chunk_total,
                "p_embedding": embedding,
                "p_confidence": confidence,
                "p_model_used": model_used,
            },
        ).execute()

        text_id = response.data

        # Update artifact processing status
        await self._update_processing_status(artifact_id, ArtifactProcessingStatus.COMPLETED)

        # Retrieve full record
        text_record = await self._get_text_by_id(text_id)
        if not text_record:
            raise RuntimeError(f"Failed to retrieve created text surrogate {text_id}")

        logger.debug(
            "Added {} text surrogate to artifact {} (id={})", text_kind.value, artifact_id, text_id
        )

        return text_record

    async def add_vision_summary(
        self,
        artifact_id: str,
        summary: str,
        is_short: bool = True,
        model_used: str | None = None,
    ) -> ArtifactText:
        """Add vision summary to an image artifact.

        Args:
            artifact_id: Artifact UUID.
            summary: Vision description.
            is_short: If True, stores as VISION_SUMMARY; else VISION_DETAIL.
            model_used: Optional model name.

        Returns:
            Created ArtifactText.
        """
        kind = TextSurrogateKind.VISION_SUMMARY if is_short else TextSurrogateKind.VISION_DETAIL
        return await self.add_text_surrogate(
            artifact_id=artifact_id,
            text_kind=kind,
            text_content=summary,
            model_used=model_used,
        )

    async def add_ocr_text(
        self,
        artifact_id: str,
        ocr_text: str,
        confidence: float | None = None,
        model_used: str | None = None,
    ) -> ArtifactText:
        """Add OCR text to an image artifact.

        Args:
            artifact_id: Artifact UUID.
            ocr_text: Extracted text.
            confidence: Optional OCR confidence.
            model_used: Optional OCR model/engine.

        Returns:
            Created ArtifactText.
        """
        return await self.add_text_surrogate(
            artifact_id=artifact_id,
            text_kind=TextSurrogateKind.OCR_TEXT,
            text_content=ocr_text,
            confidence=confidence,
            model_used=model_used,
        )

    async def add_document_text(
        self,
        artifact_id: str,
        extracted_text: str,
        summary: str | None = None,
        chunk_index: int | None = None,
        chunk_total: int | None = None,
        model_used: str | None = None,
    ) -> list[ArtifactText]:
        """Add extracted text and optional summary to a document artifact.

        Args:
            artifact_id: Artifact UUID.
            extracted_text: Full extracted text (or chunk).
            summary: Optional document summary.
            chunk_index: Optional chunk index.
            chunk_total: Optional total chunks.
            model_used: Optional model name.

        Returns:
            List of created ArtifactText records.
        """
        results = []

        # Add extracted text
        text_kind = (
            TextSurrogateKind.TEXT_CHUNK
            if chunk_index is not None
            else TextSurrogateKind.EXTRACTED_TEXT
        )
        text_record = await self.add_text_surrogate(
            artifact_id=artifact_id,
            text_kind=text_kind,
            text_content=extracted_text,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            model_used=model_used,
        )
        results.append(text_record)

        # Add summary if provided
        if summary and chunk_index is None:  # Only add summary for first chunk
            summary_record = await self.add_text_surrogate(
                artifact_id=artifact_id,
                text_kind=TextSurrogateKind.FILE_SUMMARY,
                text_content=summary,
                model_used=model_used,
            )
            results.append(summary_record)

        return results

    async def get_text_surrogates(
        self,
        artifact_id: str,
        text_kinds: list[TextSurrogateKind] | None = None,
    ) -> list[ArtifactText]:
        """Get text surrogates for an artifact.

        Args:
            artifact_id: Artifact UUID.
            text_kinds: Optional filter for text kinds.

        Returns:
            List of ArtifactText records.
        """
        try:
            query = (
                self._db._client.table("artifact_text").select("*").eq("artifact_id", artifact_id)
            )

            if text_kinds:
                kind_values = [k.value for k in text_kinds]
                query = query.in_("text_kind", kind_values)

            response = query.order("text_kind").order("chunk_index").execute()

            return [ArtifactText.from_row(row) for row in response.data or []]
        except Exception as e:
            logger.error("Failed to get text surrogates for {}: {}", artifact_id, e)
            return []

    async def _get_text_by_id(self, text_id: str) -> ArtifactText | None:
        """Get text surrogate by ID."""
        try:
            response = (
                self._db._client.table("artifact_text")
                .select("*")
                .eq("id", text_id)
                .maybe_single()
                .execute()
            )
            if response.data:
                return ArtifactText.from_row(response.data)
            return None
        except Exception as e:
            logger.error("Failed to get text surrogate {}: {}", text_id, e)
            return None

    # -------------------------------------------------------------------------
    # Processing Status
    # -------------------------------------------------------------------------

    async def set_processing_status(
        self,
        artifact_id: str,
        status: ArtifactProcessingStatus,
        error: str | None = None,
    ) -> None:
        """Set processing status for an artifact.

        Args:
            artifact_id: Artifact UUID.
            status: New processing status.
            error: Optional error message.
        """
        await self._update_processing_status(artifact_id, status, error)

    async def _update_processing_status(
        self,
        artifact_id: str,
        status: ArtifactProcessingStatus,
        error: str | None = None,
    ) -> None:
        """Update artifact processing status."""
        try:
            updates: dict[str, Any] = {
                "processing_status": status.value,
                "updated_at": datetime.now().isoformat(),
            }
            if error:
                updates["processing_error"] = error

            self._db._client.table("artifacts").update(updates).eq("id", artifact_id).execute()
        except Exception as e:
            logger.error("Failed to update processing status for {}: {}", artifact_id, e)

    # -------------------------------------------------------------------------
    # Context Building Support
    # -------------------------------------------------------------------------

    async def get_surrogates_for_context(
        self,
        episode_id: str,
        max_per_artifact: int = 2,
        max_total: int = 5,
    ) -> list[TextSurrogateForContext]:
        """Get text surrogates formatted for context building.

        This is the integration point with ContextBuilder. Returns prioritized
        text surrogates from artifacts in the episode.

        Args:
            episode_id: Episode UUID.
            max_per_artifact: Max surrogates per artifact.
            max_total: Max total surrogates to return.

        Returns:
            List of TextSurrogateForContext for inclusion in LLM context.
        """
        try:
            response = self._db._client.rpc(
                "get_artifact_surrogates_for_context",
                {
                    "p_episode_id": episode_id,
                    "p_max_per_artifact": max_per_artifact,
                    "p_max_total": max_total,
                },
            ).execute()

            results = []
            for row in response.data or []:
                results.append(
                    TextSurrogateForContext(
                        artifact_id=str(row["artifact_id"]),
                        artifact_type=ArtifactType(row["artifact_type"]),
                        original_filename=row.get("original_filename"),
                        text_kind=TextSurrogateKind(row["text_kind"]),
                        text_content=row["text_content"],
                        chunk_info=row.get("chunk_info"),
                    )
                )

            return results
        except Exception as e:
            logger.error("Failed to get surrogates for context: {}", e)
            return []


# Global instance for dependency injection
_artifact_service: ArtifactService | None = None


def get_artifact_service() -> ArtifactService:
    """Get or create global ArtifactService instance.

    Returns:
        ArtifactService instance.
    """
    global _artifact_service
    if _artifact_service is None:
        _artifact_service = ArtifactService()
    return _artifact_service


def set_artifact_service(service: ArtifactService | None) -> None:
    """Set global ArtifactService instance (useful for testing).

    Args:
        service: ArtifactService instance or None to reset.
    """
    global _artifact_service
    _artifact_service = service
