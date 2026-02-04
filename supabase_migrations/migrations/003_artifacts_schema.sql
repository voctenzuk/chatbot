-- ============================================
-- Migration: 003_artifacts_schema
-- Description: Artifacts pipeline for file storage with text surrogates
--              Supports images (vision_summary, OCR) and documents (extracted text, summary)
--              Uses sha256 for deduplication
-- See: ARCHITECTURE/MEMORY_DESIGN.md (Artifacts section)
-- ============================================

-- Enable pgvector extension for artifact text embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- 1. ARTIFACTS TABLE
-- Stores file metadata with sha256 deduplication
-- ============================================
CREATE TABLE artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    -- Ownership and context
    user_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    thread_id UUID REFERENCES threads(id) ON DELETE SET NULL,
    episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
    message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    
    -- File metadata
    type TEXT NOT NULL CHECK (type IN ('image', 'document', 'audio', 'video', 'other')),
    mime_type TEXT NOT NULL,
    original_filename TEXT,
    size_bytes BIGINT NOT NULL,
    sha256 TEXT NOT NULL,  -- For deduplication
    
    -- Storage reference (local path or S3/Supabase key)
    storage_key TEXT NOT NULL,  -- e.g., "artifacts/{user_id}/{artifact_id}/{filename}"
    storage_provider TEXT NOT NULL DEFAULT 'local',  -- 'local', 's3', 'supabase'
    
    -- Status
    processing_status TEXT NOT NULL DEFAULT 'pending' 
        CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    processing_error TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for artifacts
CREATE INDEX idx_artifacts_user_id ON artifacts(user_id);
CREATE INDEX idx_artifacts_thread_id ON artifacts(thread_id) WHERE thread_id IS NOT NULL;
CREATE INDEX idx_artifacts_episode_id ON artifacts(episode_id) WHERE episode_id IS NOT NULL;
CREATE INDEX idx_artifacts_message_id ON artifacts(message_id) WHERE message_id IS NOT NULL;
CREATE INDEX idx_artifacts_sha256 ON artifacts(sha256);  -- For deduplication lookups
CREATE INDEX idx_artifacts_type ON artifacts(type);
CREATE INDEX idx_artifacts_status ON artifacts(processing_status);
CREATE INDEX idx_artifacts_created_at ON artifacts(created_at);

-- Composite index for common query: user's artifacts in an episode
CREATE INDEX idx_artifacts_user_episode ON artifacts(user_id, episode_id) WHERE episode_id IS NOT NULL;

-- ============================================
-- 2. ARTIFACT_TEXT TABLE
-- Text surrogates and extracted content from artifacts
-- Supports vision summaries, OCR, document extraction, chunking
-- ============================================
CREATE TABLE artifact_text (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    
    -- Text kind: what type of extracted/surrogate text this is
    text_kind TEXT NOT NULL CHECK (
        text_kind IN (
            'user_caption',       -- User-provided caption/description
            'vision_summary',     -- Short vision description (1-2 lines)
            'vision_detail',      -- Detailed vision description (paragraph)
            'ocr_text',           -- OCR extracted text from image
            'file_summary',       -- Document summary
            'extracted_text',     -- Full extracted text from document
            'text_chunk',         -- Chunk of extracted text (for long docs)
            'transcript',         -- Audio/video transcript
            'auto_caption'        -- Auto-generated caption
        )
    ),
    
    -- Content
    text_content TEXT NOT NULL,
    
    -- Chunking support (for long documents)
    chunk_index INTEGER NULL,  -- Position in document (0-indexed)
    chunk_total INTEGER NULL,  -- Total chunks for this artifact
    
    -- Optional embedding for semantic search (pgvector)
    -- Using 1536 dimensions (OpenAI text-embedding-3-small)
    embedding VECTOR(1536) NULL,
    
    -- Metadata
    confidence FLOAT NULL,  -- Confidence score (0.0-1.0), e.g., for OCR
    model_used TEXT NULL,   -- Model used for extraction (e.g., "gpt-4-vision", "tesseract")
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for artifact_text
CREATE INDEX idx_artifact_text_artifact_id ON artifact_text(artifact_id);
CREATE INDEX idx_artifact_text_kind ON artifact_text(text_kind);
CREATE INDEX idx_artifact_text_artifact_kind ON artifact_text(artifact_id, text_kind);

-- GIN index for text search (fallback to semantic search)
CREATE INDEX idx_artifact_text_content ON artifact_text USING GIN (to_tsvector('english', text_content));

-- HNSW index for vector similarity search (pgvector)
-- Note: HNSW is faster but uses more memory. For large scale, consider IVFFlat.
CREATE INDEX idx_artifact_text_embedding ON artifact_text 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================
-- 3. UNIQUE CONSTRAINTS
-- Prevent duplicate text kinds per artifact (except chunks)
-- ============================================
-- Unique constraint for non-chunk text kinds (one per artifact)
CREATE UNIQUE INDEX idx_artifact_text_unique_kind 
    ON artifact_text (artifact_id, text_kind) 
    WHERE chunk_index IS NULL;

-- Unique constraint for chunks (one per index per artifact)
CREATE UNIQUE INDEX idx_artifact_text_unique_chunk 
    ON artifact_text (artifact_id, text_kind, chunk_index) 
    WHERE chunk_index IS NOT NULL;

-- ============================================
-- ENABLE ROW LEVEL SECURITY
-- ============================================
ALTER TABLE artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE artifact_text ENABLE ROW LEVEL SECURITY;

-- ============================================
-- RLS POLICIES - SERVICE ROLE (Full access)
-- ============================================
CREATE POLICY "Service role full access on artifacts"
ON artifacts FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access on artifact_text"
ON artifact_text FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- ============================================
-- RLS POLICIES - USER (Access own data only)
-- ============================================

-- Users can read their own artifacts
CREATE POLICY "Users can read own artifacts"
ON artifacts FOR SELECT
TO authenticated
USING (user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can create their own artifacts
CREATE POLICY "Users can create own artifacts"
ON artifacts FOR INSERT
TO authenticated
WITH CHECK (user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can update their own artifacts (e.g., processing status)
CREATE POLICY "Users can update own artifacts"
ON artifacts FOR UPDATE
TO authenticated
USING (user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT))
WITH CHECK (user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can delete their own artifacts
CREATE POLICY "Users can delete own artifacts"
ON artifacts FOR DELETE
TO authenticated
USING (user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can read artifact_text for their artifacts
CREATE POLICY "Users can read own artifact_text"
ON artifact_text FOR SELECT
TO authenticated
USING (
    EXISTS (
        SELECT 1 FROM artifacts
        WHERE artifacts.id = artifact_text.artifact_id
        AND artifacts.user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
);

-- Users can create artifact_text for their artifacts
CREATE POLICY "Users can create artifact_text for own artifacts"
ON artifact_text FOR INSERT
TO authenticated
WITH CHECK (
    EXISTS (
        SELECT 1 FROM artifacts
        WHERE artifacts.id = artifact_text.artifact_id
        AND artifacts.user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
);

-- Users can update artifact_text for their artifacts
CREATE POLICY "Users can update own artifact_text"
ON artifact_text FOR UPDATE
TO authenticated
USING (
    EXISTS (
        SELECT 1 FROM artifacts
        WHERE artifacts.id = artifact_text.artifact_id
        AND artifacts.user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
)
WITH CHECK (
    EXISTS (
        SELECT 1 FROM artifacts
        WHERE artifacts.id = artifact_text.artifact_id
        AND artifacts.user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to check if artifact exists by sha256 (for deduplication)
CREATE OR REPLACE FUNCTION get_artifact_by_sha256(
    p_user_id BIGINT,
    p_sha256 TEXT
)
RETURNS TABLE (
    artifact_id UUID,
    storage_key TEXT,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        a.id AS artifact_id,
        a.storage_key,
        a.created_at
    FROM artifacts a
    WHERE a.user_id = p_user_id
      AND a.sha256 = p_sha256
    LIMIT 1;
END;
$$;

-- Function to get artifacts for an episode with their text surrogates
CREATE OR REPLACE FUNCTION get_artifacts_for_episode(
    p_episode_id UUID,
    p_text_kinds TEXT[] DEFAULT NULL  -- Filter by text kinds, NULL for all
)
RETURNS TABLE (
    artifact_id UUID,
    artifact_type TEXT,
    mime_type TEXT,
    original_filename TEXT,
    storage_key TEXT,
    processing_status TEXT,
    created_at TIMESTAMPTZ,
    text_surrogates JSONB  -- Array of text surrogate objects
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        a.id AS artifact_id,
        a.type AS artifact_type,
        a.mime_type,
        a.original_filename,
        a.storage_key,
        a.processing_status,
        a.created_at,
        COALESCE(
            JSONB_AGG(
                JSONB_BUILD_OBJECT(
                    'text_kind', at.text_kind,
                    'text_content', at.text_content,
                    'chunk_index', at.chunk_index,
                    'chunk_total', at.chunk_total,
                    'confidence', at.confidence,
                    'model_used', at.model_used
                ) ORDER BY at.text_kind, at.chunk_index NULLS FIRST
            ) FILTER (WHERE at.id IS NOT NULL),
            '[]'::jsonb
        ) AS text_surrogates
    FROM artifacts a
    LEFT JOIN artifact_text at ON at.artifact_id = a.id
        AND (p_text_kinds IS NULL OR at.text_kind = ANY(p_text_kinds))
    WHERE a.episode_id = p_episode_id
    GROUP BY a.id, a.type, a.mime_type, a.original_filename, a.storage_key, a.processing_status, a.created_at
    ORDER BY a.created_at DESC;
END;
$$;

-- Function to add artifact with deduplication check
CREATE OR REPLACE FUNCTION add_artifact(
    p_user_id BIGINT,
    p_type TEXT,
    p_mime_type TEXT,
    p_size_bytes BIGINT,
    p_sha256 TEXT,
    p_storage_key TEXT,
    p_storage_provider TEXT DEFAULT 'local',
    p_original_filename TEXT DEFAULT NULL,
    p_thread_id UUID DEFAULT NULL,
    p_episode_id UUID DEFAULT NULL,
    p_message_id UUID DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_artifact_id UUID;
BEGIN
    -- Check for existing artifact with same sha256 for this user
    SELECT id INTO v_artifact_id
    FROM artifacts
    WHERE user_id = p_user_id AND sha256 = p_sha256
    LIMIT 1;
    
    -- If exists, return existing artifact ID (deduplication)
    IF v_artifact_id IS NOT NULL THEN
        -- Optionally update context references if they changed
        UPDATE artifacts
        SET 
            thread_id = COALESCE(p_thread_id, thread_id),
            episode_id = COALESCE(p_episode_id, episode_id),
            message_id = COALESCE(p_message_id, message_id),
            updated_at = NOW()
        WHERE id = v_artifact_id;
        
        RETURN v_artifact_id;
    END IF;
    
    -- Insert new artifact
    INSERT INTO artifacts (
        user_id, type, mime_type, size_bytes, sha256,
        storage_key, storage_provider, original_filename,
        thread_id, episode_id, message_id
    ) VALUES (
        p_user_id, p_type, p_mime_type, p_size_bytes, p_sha256,
        p_storage_key, p_storage_provider, p_original_filename,
        p_thread_id, p_episode_id, p_message_id
    )
    RETURNING id INTO v_artifact_id;
    
    RETURN v_artifact_id;
END;
$$;

-- Function to upsert artifact text (handles both insert and update)
CREATE OR REPLACE FUNCTION upsert_artifact_text(
    p_artifact_id UUID,
    p_text_kind TEXT,
    p_text_content TEXT,
    p_chunk_index INTEGER DEFAULT NULL,
    p_chunk_total INTEGER DEFAULT NULL,
    p_embedding VECTOR(1536) DEFAULT NULL,
    p_confidence FLOAT DEFAULT NULL,
    p_model_used TEXT DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_text_id UUID;
BEGIN
    -- Try to update existing text of same kind (and chunk if applicable)
    IF p_chunk_index IS NOT NULL THEN
        UPDATE artifact_text
        SET 
            text_content = p_text_content,
            chunk_total = p_chunk_total,
            embedding = p_embedding,
            confidence = p_confidence,
            model_used = p_model_used,
            created_at = NOW()
        WHERE artifact_id = p_artifact_id 
          AND text_kind = p_text_kind
          AND chunk_index = p_chunk_index
        RETURNING id INTO v_text_id;
    ELSE
        UPDATE artifact_text
        SET 
            text_content = p_text_content,
            embedding = p_embedding,
            confidence = p_confidence,
            model_used = p_model_used,
            created_at = NOW()
        WHERE artifact_id = p_artifact_id 
          AND text_kind = p_text_kind
          AND chunk_index IS NULL
        RETURNING id INTO v_text_id;
    END IF;
    
    -- If no existing text, insert new
    IF v_text_id IS NULL THEN
        INSERT INTO artifact_text (
            artifact_id, text_kind, text_content,
            chunk_index, chunk_total, embedding,
            confidence, model_used
        ) VALUES (
            p_artifact_id, p_text_kind, p_text_content,
            p_chunk_index, p_chunk_total, p_embedding,
            p_confidence, p_model_used
        )
        RETURNING id INTO v_text_id;
    END IF;
    
    RETURN v_text_id;
END;
$$;

-- Function to search artifact text by semantic similarity
CREATE OR REPLACE FUNCTION search_artifact_text(
    p_user_id BIGINT,
    p_query_embedding VECTOR(1536),
    p_limit INTEGER DEFAULT 5,
    p_text_kinds TEXT[] DEFAULT NULL,
    p_min_similarity FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    artifact_id UUID,
    artifact_type TEXT,
    text_id UUID,
    text_kind TEXT,
    text_content TEXT,
    similarity FLOAT,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        a.id AS artifact_id,
        a.type AS artifact_type,
        at.id AS text_id,
        at.text_kind,
        at.text_content,
        1 - (at.embedding <=> p_query_embedding) AS similarity,
        at.created_at
    FROM artifact_text at
    JOIN artifacts a ON a.id = at.artifact_id
    WHERE a.user_id = p_user_id
      AND at.embedding IS NOT NULL
      AND (p_text_kinds IS NULL OR at.text_kind = ANY(p_text_kinds))
      AND 1 - (at.embedding <=> p_query_embedding) >= p_min_similarity
    ORDER BY at.embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$;

-- Function to get text surrogate for context building (top-k per artifact)
CREATE OR REPLACE FUNCTION get_artifact_surrogates_for_context(
    p_episode_id UUID,
    p_max_per_artifact INTEGER DEFAULT 2,
    p_max_total INTEGER DEFAULT 5
)
RETURNS TABLE (
    artifact_id UUID,
    artifact_type TEXT,
    original_filename TEXT,
    text_kind TEXT,
    text_content TEXT,
    chunk_info TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    WITH ranked_texts AS (
        SELECT 
            at.artifact_id,
            a.type AS artifact_type,
            a.original_filename,
            at.text_kind,
            at.text_content,
            at.chunk_index,
            at.chunk_total,
            ROW_NUMBER() OVER (
                PARTITION BY at.artifact_id 
                ORDER BY 
                    -- Priority order for context inclusion
                    CASE at.text_kind
                        WHEN 'vision_summary' THEN 1
                        WHEN 'file_summary' THEN 2
                        WHEN 'ocr_text' THEN 3
                        WHEN 'extracted_text' THEN 4
                        WHEN 'vision_detail' THEN 5
                        ELSE 6
                    END,
                    at.chunk_index NULLS FIRST
            ) AS rank_in_artifact
        FROM artifact_text at
        JOIN artifacts a ON a.id = at.artifact_id
        WHERE a.episode_id = p_episode_id
          AND a.processing_status = 'completed'
          AND at.text_kind IN (
              'vision_summary', 'vision_detail', 'ocr_text',
              'file_summary', 'extracted_text', 'transcript'
          )
    )
    SELECT 
        rt.artifact_id,
        rt.artifact_type,
        rt.original_filename,
        rt.text_kind,
        rt.text_content,
        CASE 
            WHEN rt.chunk_total IS NOT NULL 
            THEN 'chunk ' || (rt.chunk_index + 1) || '/' || rt.chunk_total
            ELSE NULL
        END AS chunk_info
    FROM ranked_texts rt
    WHERE rt.rank_in_artifact <= p_max_per_artifact
    ORDER BY rt.rank_in_artifact
    LIMIT p_max_total;
END;
$$;

-- ============================================
-- GRANT PERMISSIONS
-- ============================================

-- Grant usage on functions to authenticated users
GRANT EXECUTE ON FUNCTION get_artifact_by_sha256(BIGINT, TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION get_artifacts_for_episode(UUID, TEXT[]) TO authenticated;
GRANT EXECUTE ON FUNCTION add_artifact(BIGINT, TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, TEXT, UUID, UUID, UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION upsert_artifact_text(UUID, TEXT, TEXT, INTEGER, INTEGER, VECTOR(1536), FLOAT, TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION search_artifact_text(BIGINT, VECTOR(1536), INTEGER, TEXT[], FLOAT) TO authenticated;
GRANT EXECUTE ON FUNCTION get_artifact_surrogates_for_context(UUID, INTEGER, INTEGER) TO authenticated;

-- Grant usage on functions to service_role
GRANT EXECUTE ON FUNCTION get_artifact_by_sha256(BIGINT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION get_artifacts_for_episode(UUID, TEXT[]) TO service_role;
GRANT EXECUTE ON FUNCTION add_artifact(BIGINT, TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, TEXT, UUID, UUID, UUID) TO service_role;
GRANT EXECUTE ON FUNCTION upsert_artifact_text(UUID, TEXT, TEXT, INTEGER, INTEGER, VECTOR(1536), FLOAT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION search_artifact_text(BIGINT, VECTOR(1536), INTEGER, TEXT[], FLOAT) TO service_role;
GRANT EXECUTE ON FUNCTION get_artifact_surrogates_for_context(UUID, INTEGER, INTEGER) TO service_role;
