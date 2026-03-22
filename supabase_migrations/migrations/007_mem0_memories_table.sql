-- mem0 vector memory table for Supabase pgvector
-- Requires: pgvector extension enabled (Supabase Dashboard → Extensions → vector)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for fast vector similarity search (cosine distance)
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_memories_metadata ON memories USING GIN (metadata);

-- Row Level Security: bot uses service_role key (bypasses RLS),
-- but enable RLS to prevent accidental exposure via anon/authenticated keys
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;

-- Service role policy: full access for the bot backend
CREATE POLICY "service_role_full_access" ON memories
    FOR ALL
    USING (auth.role() = 'service_role');

-- RPC function for mem0 vector similarity search
CREATE OR REPLACE FUNCTION match_vectors(
    query_embedding vector(1536),
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}'
)
RETURNS TABLE (
    id TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.metadata,
        1 - (m.embedding <=> query_embedding) AS similarity
    FROM memories m
    WHERE (filter = '{}' OR m.metadata @> filter)
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
