"""Storage backend abstraction for artifacts.

This module provides an abstract interface for artifact storage,
with implementations for local filesystem and future S3/Supabase support.
"""

from __future__ import annotations

import hashlib
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from loguru import logger


@dataclass
class StorageReference:
    """Reference to a stored artifact."""

    storage_key: str
    provider: str
    size_bytes: int
    sha256: str


class StorageBackend(ABC):
    """Abstract base class for artifact storage backends."""

    @abstractmethod
    async def store(
        self,
        data: bytes | BinaryIO,
        user_id: int,
        artifact_id: str,
        filename: str,
    ) -> StorageReference:
        """Store artifact data and return storage reference.

        Args:
            data: Binary data or file-like object to store.
            user_id: Telegram user ID (for organization).
            artifact_id: UUID of the artifact record.
            filename: Original filename.

        Returns:
            StorageReference with storage key and metadata.
        """
        ...

    @abstractmethod
    async def retrieve(self, storage_key: str) -> bytes | None:
        """Retrieve artifact data by storage key.

        Args:
            storage_key: Key returned by store().

        Returns:
            Binary data or None if not found.
        """
        ...

    @abstractmethod
    async def delete(self, storage_key: str) -> bool:
        """Delete artifact data.

        Args:
            storage_key: Key returned by store().

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def exists(self, storage_key: str) -> bool:
        """Check if artifact exists.

        Args:
            storage_key: Key returned by store().

        Returns:
            True if exists, False otherwise.
        """
        ...

    @abstractmethod
    def get_public_url(self, storage_key: str) -> str | None:
        """Get public URL for artifact (if supported).

        Args:
            storage_key: Key returned by store().

        Returns:
            Public URL or None if not publicly accessible.
        """
        ...

    def compute_sha256(self, data: bytes | BinaryIO) -> str:
        """Compute SHA256 hash of data.

        Args:
            data: Binary data or file-like object.

        Returns:
            Hex-encoded SHA256 hash.
        """
        if isinstance(data, bytes):
            return hashlib.sha256(data).hexdigest()
        else:
            # Read from file-like object
            hasher = hashlib.sha256()
            while chunk := data.read(8192):
                hasher.update(chunk)
            # Reset file pointer
            data.seek(0)
            return hasher.hexdigest()


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend for artifacts.

    Stores files in a directory structure: {base_path}/{user_id}/{artifact_id}/{filename}
    """

    def __init__(self, base_path: str | Path = "./data/artifacts") -> None:
        """Initialize local storage backend.

        Args:
            base_path: Base directory for artifact storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("LocalStorageBackend initialized at {}", self.base_path.absolute())

    def _get_file_path(self, storage_key: str) -> Path:
        """Convert storage key to filesystem path.

        Args:
            storage_key: Relative path from base.

        Returns:
            Absolute Path object.

        Raises:
            ValueError: If storage_key attempts path traversal.
        """
        # Security: ensure path doesn't escape base_path
        # First, reject any path components that are parent references
        if ".." in storage_key:
            raise ValueError(f"Invalid storage key (path traversal attempt): {storage_key}")

        path = (self.base_path / storage_key).resolve()

        # Verify resolved path is still within base_path
        try:
            path.relative_to(self.base_path.resolve())
        except ValueError as e:
            raise ValueError(f"Invalid storage key (path traversal attempt): {storage_key}") from e

        return path

    async def store(
        self,
        data: bytes | BinaryIO,
        user_id: int,
        artifact_id: str,
        filename: str,
    ) -> StorageReference:
        """Store artifact in local filesystem.

        Args:
            data: Binary data or file-like object.
            user_id: Telegram user ID.
            artifact_id: UUID of the artifact.
            filename: Original filename.

        Returns:
            StorageReference with relative path as key.
        """
        # Compute hash for deduplication
        sha256 = self.compute_sha256(data)

        # Create directory structure
        storage_key = f"{user_id}/{artifact_id}/{filename}"
        file_path = self._get_file_path(storage_key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        if isinstance(data, bytes):
            file_path.write_bytes(data)
            size_bytes = len(data)
        else:
            # Copy from file-like object
            with file_path.open("wb") as f:
                shutil.copyfileobj(data, f)
            size_bytes = file_path.stat().st_size

        logger.debug(
            "Stored artifact: user={}, artifact={}, size={} bytes", user_id, artifact_id, size_bytes
        )

        return StorageReference(
            storage_key=storage_key,
            provider="local",
            size_bytes=size_bytes,
            sha256=sha256,
        )

    async def retrieve(self, storage_key: str) -> bytes | None:
        """Retrieve artifact from local filesystem.

        Args:
            storage_key: Relative path from base.

        Returns:
            File contents or None if not found.
        """
        try:
            file_path = self._get_file_path(storage_key)
            if not file_path.exists():
                return None
            return file_path.read_bytes()
        except (IOError, OSError) as e:
            logger.error("Failed to retrieve artifact {}: {}", storage_key, e)
            return None

    async def delete(self, storage_key: str) -> bool:
        """Delete artifact from local filesystem.

        Args:
            storage_key: Relative path from base.

        Returns:
            True if deleted, False if not found.
        """
        try:
            file_path = self._get_file_path(storage_key)
            if not file_path.exists():
                return False
            file_path.unlink()

            # Clean up empty parent directories
            parent = file_path.parent
            while parent != self.base_path and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent

            return True
        except (IOError, OSError) as e:
            logger.error("Failed to delete artifact {}: {}", storage_key, e)
            return False

    async def exists(self, storage_key: str) -> bool:
        """Check if artifact exists.

        Args:
            storage_key: Relative path from base.

        Returns:
            True if exists.
        """
        try:
            file_path = self._get_file_path(storage_key)
            return file_path.exists()
        except (IOError, OSError, ValueError):
            return False

    def get_public_url(self, storage_key: str) -> str | None:
        """Get public URL for artifact.

        Note: Local storage doesn't provide public URLs by default.
        For local development, returns None.

        Args:
            storage_key: Key returned by store().

        Returns:
            None (local files aren't publicly accessible).
        """
        return None


class S3StorageBackend(StorageBackend):
    """S3-compatible storage backend (placeholder for future implementation).

    This is a stub implementation. Full S3 support would require boto3/aiohttp.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ) -> None:
        """Initialize S3 storage backend.

        Args:
            bucket: S3 bucket name.
            region: AWS region.
            endpoint_url: Optional endpoint URL (for MinIO, etc.).
        """
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url
        self._initialized = False
        logger.warning("S3StorageBackend is a stub - full implementation pending")

    async def _ensure_initialized(self) -> None:
        """Ensure S3 client is initialized."""
        if not self._initialized:
            raise NotImplementedError("S3 backend not yet implemented. Use LocalStorageBackend.")

    async def store(
        self,
        data: bytes | BinaryIO,
        user_id: int,
        artifact_id: str,
        filename: str,
    ) -> StorageReference:
        """Store artifact in S3 (not implemented)."""
        await self._ensure_initialized()
        raise NotImplementedError()

    async def retrieve(self, storage_key: str) -> bytes | None:
        """Retrieve artifact from S3 (not implemented)."""
        await self._ensure_initialized()
        raise NotImplementedError()

    async def delete(self, storage_key: str) -> bool:
        """Delete artifact from S3 (not implemented)."""
        await self._ensure_initialized()
        raise NotImplementedError()

    async def exists(self, storage_key: str) -> bool:
        """Check if artifact exists in S3 (not implemented)."""
        await self._ensure_initialized()
        raise NotImplementedError()

    def get_public_url(self, storage_key: str) -> str | None:
        """Get public URL for S3 object (not implemented)."""
        return None


# Global storage backend instance
_storage_backend: StorageBackend | None = None


def get_storage_backend() -> StorageBackend:
    """Get or create global storage backend instance.

    Returns:
        StorageBackend instance.

    Default is LocalStorageBackend. Set ARTIFACT_STORAGE_PROVIDER env var
    to change provider ('local', 's3').
    """
    global _storage_backend
    if _storage_backend is None:
        import os

        provider = os.getenv("ARTIFACT_STORAGE_PROVIDER", "local").lower()

        if provider == "s3":
            bucket = os.getenv("ARTIFACT_S3_BUCKET", "artifacts")
            region = os.getenv("ARTIFACT_S3_REGION", "us-east-1")
            endpoint = os.getenv("ARTIFACT_S3_ENDPOINT")
            _storage_backend = S3StorageBackend(bucket, region, endpoint)
        else:
            base_path = os.getenv("ARTIFACT_LOCAL_PATH", "./data/artifacts")
            _storage_backend = LocalStorageBackend(base_path)

    return _storage_backend


def set_storage_backend(backend: StorageBackend | None) -> None:
    """Set global storage backend instance (useful for testing).

    Args:
        backend: StorageBackend instance or None to reset.
    """
    global _storage_backend
    _storage_backend = backend
