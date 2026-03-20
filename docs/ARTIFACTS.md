# Artifacts Pipeline Documentation

This document describes the artifacts pipeline implemented in Epic #6, which provides file storage and text surrogate generation for the chatbot.

## Overview

The artifacts pipeline allows the bot to:
- Store files (images, documents, audio, video) with deduplication
- Generate text surrogates (vision summaries, OCR, document extraction)
- Include artifact information in conversation context

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ArtifactService                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Creation   │  │  Storage    │  │   Text Surrogates       │  │
│  │  + Dedupe   │  │  Backend    │  │   (Vision/OCR/Extract)  │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Database (Supabase)                      │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐   │
│  │   artifacts     │  │         artifact_text               │   │
│  │   (metadata)    │  │  (vision_summary, ocr_text, etc.)   │   │
│  └─────────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Storage Backend                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │    Local     │  │     S3       │  │     Supabase         │   │
│  │  Filesystem  │  │  (future)    │  │    (future)          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Database Schema

### artifacts Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `user_id` | BIGINT | Telegram user ID (foreign key) |
| `thread_id` | UUID | Optional thread reference |
| `episode_id` | UUID | Optional episode reference |
| `message_id` | UUID | Optional message reference |
| `type` | TEXT | Artifact type: `image`, `document`, `audio`, `video`, `other` |
| `mime_type` | TEXT | MIME type of the file |
| `original_filename` | TEXT | Original filename |
| `size_bytes` | BIGINT | File size in bytes |
| `sha256` | TEXT | SHA256 hash for deduplication |
| `storage_key` | TEXT | Path/key in storage backend |
| `storage_provider` | TEXT | Storage provider: `local`, `s3`, `supabase` |
| `processing_status` | TEXT | `pending`, `processing`, `completed`, `failed` |
| `processing_error` | TEXT | Error message if processing failed |

### artifact_text Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `artifact_id` | UUID | Foreign key to artifacts |
| `text_kind` | TEXT | Type of surrogate (see below) |
| `text_content` | TEXT | The extracted/surrogate text |
| `chunk_index` | INTEGER | Position in chunked document |
| `chunk_total` | INTEGER | Total chunks for document |
| `embedding` | VECTOR(1536) | Optional embedding for semantic search |
| `confidence` | FLOAT | Confidence score (0.0-1.0) |
| `model_used` | TEXT | Model/engine used for extraction |

### Text Surrogate Kinds

For **images**:
- `vision_summary` - Short description (1-2 lines)
- `vision_detail` - Detailed description (paragraph)
- `ocr_text` - Extracted text via OCR

For **documents**:
- `file_summary` - Document summary
- `extracted_text` - Full extracted text
- `text_chunk` - Chunk of extracted text

For **audio/video**:
- `transcript` - Speech-to-text transcript

For **all**:
- `user_caption` - User-provided description
- `auto_caption` - Auto-generated caption

## Usage

### Creating an Artifact

```python
from bot.services.artifact_service import (
    ArtifactService,
    CreateArtifactRequest,
    ArtifactType,
)

service = ArtifactService()

# Create from bytes
request = CreateArtifactRequest(
    user_id=12345,
    type=ArtifactType.IMAGE,
    mime_type="image/jpeg",
    filename="photo.jpg",
    data=b"...image bytes...",
    episode_id="episode-uuid",  # Optional context
)

result = await service.create_artifact(request)
artifact = result.artifact
is_duplicate = result.is_duplicate  # True if deduplicated
```

### Adding Vision Summary

```python
# After processing image with vision model
await service.add_vision_summary(
    artifact_id=artifact.id,
    summary="A photo of a ginger cat sleeping on a windowsill",
    is_short=True,  # True for VISION_SUMMARY, False for VISION_DETAIL
    model_used="gpt-4-vision",
)
```

### Adding OCR Text

```python
await service.add_ocr_text(
    artifact_id=artifact.id,
    ocr_text="Extracted text from image",
    confidence=0.95,
    model_used="tesseract",
)
```

### Adding Document Text

```python
# For short documents
await service.add_document_text(
    artifact_id=artifact.id,
    extracted_text="Full document text...",
    summary="Document about cats",
)

# For long documents (chunked)
await service.add_document_text(
    artifact_id=artifact.id,
    extracted_text="Chunk of text...",
    chunk_index=0,
    chunk_total=5,
)
```

### Context Building Integration

```python
from bot.services.artifact_service import get_artifact_service
from bot.services.context_builder import get_context_builder

artifact_service = get_artifact_service()
context_builder = get_context_builder()

# Get surrogates for current episode
surrogates = await artifact_service.get_surrogates_for_context(
    episode_id="episode-uuid",
    max_per_artifact=2,
    max_total=5,
)

# Include in context
context = context_builder.assemble(
    summary=running_summary,
    artifact_surrogates=surrogates,
    recent_messages=recent_messages,
)
```

## Storage Backends

### Local Filesystem (Default)

```python
from bot.services.storage_backend import LocalStorageBackend, get_storage_backend

# Default: stores in ./data/artifacts
storage = get_storage_backend()

# Or specify custom path
storage = LocalStorageBackend(base_path="/var/artifacts")
```

Configuration via environment:
```bash
ARTIFACT_STORAGE_PROVIDER=local
ARTIFACT_LOCAL_PATH=./data/artifacts
```

### S3 (Placeholder)

S3 backend is defined but not fully implemented. When implemented:

```bash
ARTIFACT_STORAGE_PROVIDER=s3
ARTIFACT_S3_BUCKET=my-bucket
ARTIFACT_S3_REGION=us-east-1
ARTIFACT_S3_ENDPOINT=https://s3.example.com  # Optional, for MinIO
```

## Deduplication

Artifacts are deduplicated per user using SHA256 hash:

1. When creating an artifact, compute SHA256 hash
2. Check if artifact with same hash exists for user
3. If exists, return existing artifact (updates context refs)
4. If not, store file and create new record

This prevents storing the same file multiple times while maintaining context references.

## Processing Pipeline

```
User uploads file
       │
       ▼
Compute SHA256 ──► Check for existing ──► If exists, return existing
       │                                      │
       │ No existing                          │
       ▼                                      │
Store in backend                              │
       │                                      │
       ▼                                      │
Create DB record                              │
       │◄─────────────────────────────────────┘
       ▼
Set status = PENDING
       │
       ▼
Process (async):
   - Images: Vision analysis + OCR
   - Documents: Text extraction + summarization
   - Audio/Video: Transcription
       │
       ▼
Set status = COMPLETED (or FAILED)
```

## Security

### Path Traversal Protection

Local storage validates paths to prevent traversal attacks:

```python
# Raises ValueError
backend._get_file_path("../../../etc/passwd")
```

### RLS Policies

Row Level Security ensures users can only access their own artifacts:

- `artifacts`: User can CRUD only their records
- `artifact_text`: User can access only text for their artifacts

### File Size Limits

Recommended to implement at handler level before calling ArtifactService:

```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if len(file_data) > MAX_FILE_SIZE:
    raise ValueError("File too large")
```

## Testing

Run artifact tests:

```bash
pytest tests/test_artifact_service.py -v
```

Run context builder tests (includes surrogate tests):

```bash
pytest tests/test_context_builder.py -v
```

## Future Enhancements

1. **Full S3/Supabase Storage**: Complete the S3 backend implementation
2. **Async Processing**: Background jobs for vision/OCR/document processing
3. **Vector Search**: Enable semantic search over artifact text
4. **Image Generation**: Store generated images with prompt metadata
5. **TTL/Cleanup**: Automatic cleanup of old/unused artifacts
6. **Content Moderation**: Safety checks for uploaded content

## Migration

Apply the migration to Supabase:

```bash
psql $DATABASE_URL -f supabase_migrations/migrations/003_artifacts_schema.sql
```

Or use Supabase CLI:

```bash
supabase db push
```

## API Reference

See module docstrings for detailed API documentation:
- `bot.services.artifact_service` - Artifact management
- `bot.services.storage_backend` - Storage abstraction
- `bot.services.context_builder` - Context building with surrogates
