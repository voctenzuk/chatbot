-- ============================================
-- Migration: 006_rpc_returns_full_rows
-- Description: Change RPC functions to RETURNS TABLE instead of RETURNS UUID
--              to avoid RLS issues on follow-up SELECT queries.
--              Also auto-provisions user in get_or_create_thread.
-- Depends on: 002_memory_schema, 005_fix_get_or_create_thread_auto_provision
-- ============================================

-- Must drop old functions first (return type change not allowed by CREATE OR REPLACE)
DROP FUNCTION IF EXISTS get_or_create_thread(BIGINT);
DROP FUNCTION IF EXISTS start_new_episode(UUID, TEXT);
DROP FUNCTION IF EXISTS add_message_to_current_episode(BIGINT, TEXT, TEXT, INTEGER, INTEGER, TEXT);

-- get_or_create_thread: returns full thread row
CREATE OR REPLACE FUNCTION get_or_create_thread(p_telegram_user_id BIGINT)
RETURNS TABLE (
    id UUID,
    telegram_user_id BIGINT,
    active_episode_id UUID,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    v_thread_id UUID;
    v_episode_id UUID;
BEGIN
    -- Ensure user exists
    INSERT INTO users (telegram_id)
    VALUES (p_telegram_user_id)
    ON CONFLICT (telegram_id) DO NOTHING;

    -- Try to find existing thread
    SELECT t.id INTO v_thread_id
    FROM threads t
    WHERE t.telegram_user_id = p_telegram_user_id
    LIMIT 1;

    -- If no thread exists, create one with an initial episode
    IF v_thread_id IS NULL THEN
        INSERT INTO threads (telegram_user_id)
        VALUES (p_telegram_user_id)
        RETURNING threads.id INTO v_thread_id;

        INSERT INTO episodes (thread_id, status, started_at, topic_label)
        VALUES (v_thread_id, 'active', NOW(), 'New Conversation')
        RETURNING episodes.id INTO v_episode_id;

        UPDATE threads
        SET active_episode_id = v_episode_id, updated_at = NOW()
        WHERE threads.id = v_thread_id;
    END IF;

    RETURN QUERY
    SELECT t.id, t.telegram_user_id, t.active_episode_id, t.created_at, t.updated_at
    FROM threads t
    WHERE t.id = v_thread_id;
END;
$$;

-- start_new_episode: returns full episode row
CREATE OR REPLACE FUNCTION start_new_episode(
    p_thread_id UUID,
    p_topic_label TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    thread_id UUID,
    status TEXT,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    topic_label TEXT,
    last_user_message_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    v_current_episode_id UUID;
    v_new_episode_id UUID;
BEGIN
    SELECT t.active_episode_id INTO v_current_episode_id
    FROM threads t WHERE t.id = p_thread_id;

    IF v_current_episode_id IS NOT NULL THEN
        UPDATE episodes
        SET status = 'closed', ended_at = NOW(), updated_at = NOW()
        WHERE episodes.id = v_current_episode_id;
    END IF;

    INSERT INTO episodes (thread_id, status, started_at, topic_label)
    VALUES (p_thread_id, 'active', NOW(), p_topic_label)
    RETURNING episodes.id INTO v_new_episode_id;

    UPDATE threads
    SET active_episode_id = v_new_episode_id, updated_at = NOW()
    WHERE threads.id = p_thread_id;

    RETURN QUERY
    SELECT e.id, e.thread_id, e.status, e.started_at, e.ended_at,
           e.topic_label, e.last_user_message_at, e.created_at, e.updated_at
    FROM episodes e
    WHERE e.id = v_new_episode_id;
END;
$$;

-- add_message_to_current_episode: returns full message row
CREATE OR REPLACE FUNCTION add_message_to_current_episode(
    p_telegram_user_id BIGINT,
    p_role TEXT,
    p_content_text TEXT,
    p_tokens_in INTEGER DEFAULT NULL,
    p_tokens_out INTEGER DEFAULT NULL,
    p_model TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    episode_id UUID,
    role TEXT,
    content_text TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    model TEXT,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    v_thread_id UUID;
    v_episode_id UUID;
    v_message_id UUID;
BEGIN
    SELECT t.id, t.active_episode_id
    INTO v_thread_id, v_episode_id
    FROM threads t
    WHERE t.telegram_user_id = p_telegram_user_id
    LIMIT 1;

    IF v_thread_id IS NULL THEN
        SELECT got.id, got.active_episode_id
        INTO v_thread_id, v_episode_id
        FROM get_or_create_thread(p_telegram_user_id) got;
    END IF;

    IF v_episode_id IS NULL THEN
        INSERT INTO episodes (thread_id, status, started_at)
        VALUES (v_thread_id, 'active', NOW())
        RETURNING episodes.id INTO v_episode_id;

        UPDATE threads
        SET active_episode_id = v_episode_id
        WHERE threads.id = v_thread_id;
    END IF;

    INSERT INTO messages (episode_id, role, content_text, tokens_in, tokens_out, model)
    VALUES (v_episode_id, p_role, p_content_text, p_tokens_in, p_tokens_out, p_model)
    RETURNING messages.id INTO v_message_id;

    IF p_role = 'user' THEN
        UPDATE episodes
        SET last_user_message_at = NOW(), updated_at = NOW()
        WHERE episodes.id = v_episode_id;
    END IF;

    RETURN QUERY
    SELECT m.id, m.episode_id, m.role, m.content_text,
           m.tokens_in, m.tokens_out, m.model, m.created_at
    FROM messages m
    WHERE m.id = v_message_id;
END;
$$;

-- Re-grant permissions after DROP + CREATE
GRANT EXECUTE ON FUNCTION get_or_create_thread(BIGINT) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION start_new_episode(UUID, TEXT) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION add_message_to_current_episode(BIGINT, TEXT, TEXT, INTEGER, INTEGER, TEXT) TO authenticated, service_role;
