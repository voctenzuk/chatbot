-- ============================================
-- Migration: 002_memory_schema
-- Description: Memory system schema for threads, episodes, messages, and summaries
-- See: ARCHITECTURE/MEMORY_DESIGN.md
-- ============================================

-- Enable uuid-ossp extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. THREADS TABLE
-- A thread represents one conversational stream per user/chat (stable over time)
-- ============================================
CREATE TABLE threads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    telegram_user_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    active_episode_id UUID NULL,  -- Will be set after episodes table is created
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for threads
CREATE INDEX idx_threads_telegram_user_id ON threads(telegram_user_id);
CREATE INDEX idx_threads_active_episode_id ON threads(active_episode_id) WHERE active_episode_id IS NOT NULL;

-- ============================================
-- 2. EPISODES TABLE
-- An episode represents one "chapter" within a thread (session/episodic memory)
-- ============================================
CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closed')),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ NULL,
    topic_label TEXT NULL,
    last_user_message_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for episodes
CREATE INDEX idx_episodes_thread_id ON episodes(thread_id);
CREATE INDEX idx_episodes_status ON episodes(status);
CREATE INDEX idx_episodes_thread_status ON episodes(thread_id, status);
CREATE INDEX idx_episodes_last_user_message_at ON episodes(last_user_message_at);

-- Now add the foreign key constraint to threads.active_episode_id
ALTER TABLE threads 
    ADD CONSTRAINT fk_threads_active_episode 
    FOREIGN KEY (active_episode_id) 
    REFERENCES episodes(id) 
    ON DELETE SET NULL;

-- ============================================
-- 3. MESSAGES TABLE (Episode Messages)
-- Raw message log for each episode
-- ============================================
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content_text TEXT NOT NULL,
    tokens_in INTEGER NULL,
    tokens_out INTEGER NULL,
    model TEXT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for messages
CREATE INDEX idx_messages_episode_id ON messages(episode_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_messages_episode_created ON messages(episode_id, created_at);

-- ============================================
-- 4. EPISODE SUMMARIES TABLE
-- Summaries of episodes: running, chunk, and final summaries
-- ============================================
CREATE TABLE episode_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('running', 'chunk', 'final')),
    summary_text TEXT NOT NULL,
    summary_json JSONB NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for episode_summaries
CREATE INDEX idx_episode_summaries_episode_id ON episode_summaries(episode_id);
CREATE INDEX idx_episode_summaries_kind ON episode_summaries(kind);
CREATE INDEX idx_episode_summaries_episode_kind ON episode_summaries(episode_id, kind);

-- GIN index for JSONB search in summary_json
CREATE INDEX idx_episode_summaries_summary_json ON episode_summaries USING GIN (summary_json jsonb_path_ops);

-- ============================================
-- ENABLE ROW LEVEL SECURITY
-- ============================================
ALTER TABLE threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE episode_summaries ENABLE ROW LEVEL SECURITY;

-- ============================================
-- RLS POLICIES - SERVICE ROLE (Full access)
-- ============================================

CREATE POLICY "Service role full access on threads"
ON threads FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access on episodes"
ON episodes FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access on messages"
ON messages FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access on episode_summaries"
ON episode_summaries FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- ============================================
-- RLS POLICIES - USER (Access own data only)
-- Filter by telegram_id from JWT claim via threads join
-- ============================================

-- Users can read their own threads
CREATE POLICY "Users can read own threads"
ON threads FOR SELECT
TO authenticated
USING (telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can create their own threads
CREATE POLICY "Users can create own threads"
ON threads FOR INSERT
TO authenticated
WITH CHECK (telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can update their own threads (e.g., set active_episode_id)
CREATE POLICY "Users can update own threads"
ON threads FOR UPDATE
TO authenticated
USING (telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT))
WITH CHECK (telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can read episodes for their threads
CREATE POLICY "Users can read own episodes"
ON episodes FOR SELECT
TO authenticated
USING (
    EXISTS (
        SELECT 1 FROM threads
        WHERE threads.id = episodes.thread_id
        AND threads.telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
);

-- Users can create episodes for their threads
CREATE POLICY "Users can create episodes for own threads"
ON episodes FOR INSERT
TO authenticated
WITH CHECK (
    EXISTS (
        SELECT 1 FROM threads
        WHERE threads.id = episodes.thread_id
        AND threads.telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
);

-- Users can update episodes for their threads
CREATE POLICY "Users can update own episodes"
ON episodes FOR UPDATE
TO authenticated
USING (
    EXISTS (
        SELECT 1 FROM threads
        WHERE threads.id = episodes.thread_id
        AND threads.telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
)
WITH CHECK (
    EXISTS (
        SELECT 1 FROM threads
        WHERE threads.id = episodes.thread_id
        AND threads.telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
);

-- Users can read messages for their episodes
CREATE POLICY "Users can read own messages"
ON messages FOR SELECT
TO authenticated
USING (
    EXISTS (
        SELECT 1 FROM episodes
        JOIN threads ON threads.id = episodes.thread_id
        WHERE episodes.id = messages.episode_id
        AND threads.telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
);

-- Users can create messages for their episodes
CREATE POLICY "Users can create messages for own episodes"
ON messages FOR INSERT
TO authenticated
WITH CHECK (
    EXISTS (
        SELECT 1 FROM episodes
        JOIN threads ON threads.id = episodes.thread_id
        WHERE episodes.id = messages.episode_id
        AND threads.telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
);

-- Users can read summaries for their episodes
CREATE POLICY "Users can read own episode summaries"
ON episode_summaries FOR SELECT
TO authenticated
USING (
    EXISTS (
        SELECT 1 FROM episodes
        JOIN threads ON threads.id = episodes.thread_id
        WHERE episodes.id = episode_summaries.episode_id
        AND threads.telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
);

-- Users can create summaries for their episodes
CREATE POLICY "Users can create summaries for own episodes"
ON episode_summaries FOR INSERT
TO authenticated
WITH CHECK (
    EXISTS (
        SELECT 1 FROM episodes
        JOIN threads ON threads.id = episodes.thread_id
        WHERE episodes.id = episode_summaries.episode_id
        AND threads.telegram_user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT)
    )
);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to get or create a thread for a user
CREATE OR REPLACE FUNCTION get_or_create_thread(p_telegram_user_id BIGINT)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_thread_id UUID;
    v_episode_id UUID;
BEGIN
    -- Try to find existing thread
    SELECT id INTO v_thread_id
    FROM threads
    WHERE telegram_user_id = p_telegram_user_id
    LIMIT 1;
    
    -- If no thread exists, create one with an initial episode
    IF v_thread_id IS NULL THEN
        -- Insert thread first (without active_episode_id)
        INSERT INTO threads (telegram_user_id)
        VALUES (p_telegram_user_id)
        RETURNING id INTO v_thread_id;
        
        -- Create initial episode
        INSERT INTO episodes (thread_id, status, started_at, topic_label)
        VALUES (v_thread_id, 'active', NOW(), 'New Conversation')
        RETURNING id INTO v_episode_id;
        
        -- Update thread with active episode
        UPDATE threads
        SET active_episode_id = v_episode_id,
            updated_at = NOW()
        WHERE id = v_thread_id;
    END IF;
    
    RETURN v_thread_id;
END;
$$;

-- Function to start a new episode
CREATE OR REPLACE FUNCTION start_new_episode(
    p_thread_id UUID,
    p_topic_label TEXT DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_current_episode_id UUID;
    v_new_episode_id UUID;
BEGIN
    -- Get current active episode
    SELECT active_episode_id INTO v_current_episode_id
    FROM threads
    WHERE id = p_thread_id;
    
    -- Close current episode if exists
    IF v_current_episode_id IS NOT NULL THEN
        UPDATE episodes
        SET status = 'closed',
            ended_at = NOW(),
            updated_at = NOW()
        WHERE id = v_current_episode_id;
    END IF;
    
    -- Create new episode
    INSERT INTO episodes (thread_id, status, started_at, topic_label)
    VALUES (p_thread_id, 'active', NOW(), p_topic_label)
    RETURNING id INTO v_new_episode_id;
    
    -- Update thread with new active episode
    UPDATE threads
    SET active_episode_id = v_new_episode_id,
        updated_at = NOW()
    WHERE id = p_thread_id;
    
    RETURN v_new_episode_id;
END;
$$;

-- Function to add a message to the current episode
CREATE OR REPLACE FUNCTION add_message_to_current_episode(
    p_telegram_user_id BIGINT,
    p_role TEXT,
    p_content_text TEXT,
    p_tokens_in INTEGER DEFAULT NULL,
    p_tokens_out INTEGER DEFAULT NULL,
    p_model TEXT DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_thread_id UUID;
    v_episode_id UUID;
    v_message_id UUID;
BEGIN
    -- Get thread and active episode
    SELECT t.id, t.active_episode_id
    INTO v_thread_id, v_episode_id
    FROM threads t
    WHERE t.telegram_user_id = p_telegram_user_id
    LIMIT 1;
    
    -- If no thread, create one
    IF v_thread_id IS NULL THEN
        v_thread_id := get_or_create_thread(p_telegram_user_id);
        
        -- Get the active episode from the new thread
        SELECT active_episode_id INTO v_episode_id
        FROM threads
        WHERE id = v_thread_id;
    END IF;
    
    -- If no active episode, create one
    IF v_episode_id IS NULL THEN
        INSERT INTO episodes (thread_id, status, started_at)
        VALUES (v_thread_id, 'active', NOW())
        RETURNING id INTO v_episode_id;
        
        UPDATE threads
        SET active_episode_id = v_episode_id
        WHERE id = v_thread_id;
    END IF;
    
    -- Insert message
    INSERT INTO messages (episode_id, role, content_text, tokens_in, tokens_out, model)
    VALUES (v_episode_id, p_role, p_content_text, p_tokens_in, p_tokens_out, p_model)
    RETURNING id INTO v_message_id;
    
    -- Update episode's last_user_message_at if user message
    IF p_role = 'user' THEN
        UPDATE episodes
        SET last_user_message_at = NOW(),
            updated_at = NOW()
        WHERE id = v_episode_id;
    END IF;
    
    RETURN v_message_id;
END;
$$;

-- Function to get recent messages for context
CREATE OR REPLACE FUNCTION get_recent_messages(
    p_telegram_user_id BIGINT,
    p_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    message_id UUID,
    episode_id UUID,
    role TEXT,
    content_text TEXT,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id AS message_id,
        m.episode_id,
        m.role,
        m.content_text,
        m.created_at
    FROM messages m
    JOIN episodes e ON e.id = m.episode_id
    JOIN threads t ON t.id = e.thread_id
    WHERE t.telegram_user_id = p_telegram_user_id
    ORDER BY m.created_at DESC
    LIMIT p_limit;
END;
$$;

-- Function to update episode summary
CREATE OR REPLACE FUNCTION upsert_episode_summary(
    p_episode_id UUID,
    p_kind TEXT,
    p_summary_text TEXT,
    p_summary_json JSONB DEFAULT NULL
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_summary_id UUID;
BEGIN
    -- Try to update existing summary of same kind
    UPDATE episode_summaries
    SET summary_text = p_summary_text,
        summary_json = p_summary_json,
        created_at = NOW()
    WHERE episode_id = p_episode_id AND kind = p_kind
    RETURNING id INTO v_summary_id;
    
    -- If no existing summary, insert new
    IF v_summary_id IS NULL THEN
        INSERT INTO episode_summaries (episode_id, kind, summary_text, summary_json)
        VALUES (p_episode_id, p_kind, p_summary_text, p_summary_json)
        RETURNING id INTO v_summary_id;
    END IF;
    
    RETURN v_summary_id;
END;
$$;

-- ============================================
-- GRANT PERMISSIONS
-- ============================================

-- Grant usage on functions to authenticated users
GRANT EXECUTE ON FUNCTION get_or_create_thread(BIGINT) TO authenticated;
GRANT EXECUTE ON FUNCTION start_new_episode(UUID, TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION add_message_to_current_episode(BIGINT, TEXT, TEXT, INTEGER, INTEGER, TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION get_recent_messages(BIGINT, INTEGER) TO authenticated;
GRANT EXECUTE ON FUNCTION upsert_episode_summary(UUID, TEXT, TEXT, JSONB) TO authenticated;

-- Grant usage on functions to service_role
GRANT EXECUTE ON FUNCTION get_or_create_thread(BIGINT) TO service_role;
GRANT EXECUTE ON FUNCTION start_new_episode(UUID, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION add_message_to_current_episode(BIGINT, TEXT, TEXT, INTEGER, INTEGER, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION get_recent_messages(BIGINT, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION upsert_episode_summary(UUID, TEXT, TEXT, JSONB) TO service_role;
