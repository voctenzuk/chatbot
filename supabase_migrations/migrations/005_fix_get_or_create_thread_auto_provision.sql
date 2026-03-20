-- ============================================
-- Migration: 005_fix_get_or_create_thread_auto_provision
-- Description: Auto-provision user in get_or_create_thread to avoid FK violation
--              when episode_manager calls before provision_user_with_free_plan
-- Depends on: 002_memory_schema
-- ============================================

CREATE OR REPLACE FUNCTION get_or_create_thread(p_telegram_user_id BIGINT)
RETURNS UUID LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    v_thread_id UUID;
    v_episode_id UUID;
BEGIN
    -- Ensure user exists (auto-provision if missing)
    INSERT INTO users (telegram_id)
    VALUES (p_telegram_user_id)
    ON CONFLICT (telegram_id) DO NOTHING;

    -- Try to find existing thread
    SELECT id INTO v_thread_id FROM threads WHERE telegram_user_id = p_telegram_user_id LIMIT 1;

    -- If no thread exists, create one with an initial episode
    IF v_thread_id IS NULL THEN
        INSERT INTO threads (telegram_user_id) VALUES (p_telegram_user_id) RETURNING id INTO v_thread_id;
        INSERT INTO episodes (thread_id, status, started_at, topic_label) VALUES (v_thread_id, 'active', NOW(), 'New Conversation') RETURNING id INTO v_episode_id;
        UPDATE threads SET active_episode_id = v_episode_id, updated_at = NOW() WHERE id = v_thread_id;
    END IF;

    RETURN v_thread_id;
END;
$$;
