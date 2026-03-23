-- Migration 011: user_facts table + RPCs for fact extraction pipeline
-- Idempotent: CREATE TABLE IF NOT EXISTS, CREATE OR REPLACE FUNCTION

CREATE TABLE IF NOT EXISTS user_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id BIGINT NOT NULL REFERENCES users(telegram_id),
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    category TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance FLOAT NOT NULL DEFAULT 1.0,
    emotional_valence FLOAT DEFAULT 0.0,
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_user_facts_user ON user_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_user_facts_category ON user_facts(user_id, category);

-- RPC for fact signal aggregation (used by RelationshipScorer in Phase 2)
CREATE OR REPLACE FUNCTION get_user_facts_summary(p_user_id BIGINT)
RETURNS TABLE(personal_disclosures BIGINT, relationship_memories BIGINT, avg_emotional_depth FLOAT)
LANGUAGE sql STABLE AS $$
    SELECT
        COUNT(*) FILTER (WHERE category IN ('semantic', 'emotional')) AS personal_disclosures,
        COUNT(*) FILTER (WHERE memory_type IN ('milestone', 'inside_joke', 'boundary')) AS relationship_memories,
        COALESCE(AVG(ABS(emotional_valence)), 0.0) AS avg_emotional_depth
    FROM user_facts
    WHERE user_id = p_user_id;
$$;

-- RPC for message stats with consecutive_days via gaps-and-islands
CREATE OR REPLACE FUNCTION get_message_stats(p_user_id BIGINT)
RETURNS TABLE(total_messages BIGINT, days_active BIGINT, consecutive_days BIGINT, days_since_last BIGINT)
LANGUAGE sql STABLE AS $$
    WITH user_dates AS (
        SELECT DISTINCT DATE(created_at) AS d
        FROM messages
        WHERE telegram_user_id = p_user_id AND role = 'user'
    ),
    streak AS (
        SELECT d, d + (ROW_NUMBER() OVER (ORDER BY d DESC))::int * INTERVAL '1 day' AS grp
        FROM user_dates
    ),
    today_group AS (
        SELECT grp FROM streak WHERE d = CURRENT_DATE LIMIT 1
    )
    SELECT
        (SELECT COUNT(*) FROM messages WHERE telegram_user_id = p_user_id AND role = 'user') AS total_messages,
        (SELECT COUNT(*) FROM user_dates) AS days_active,
        COALESCE((SELECT COUNT(*) FROM streak WHERE grp = (SELECT grp FROM today_group)), 0) AS consecutive_days,
        COALESCE((SELECT CURRENT_DATE - MAX(d) FROM user_dates), 0)::BIGINT AS days_since_last;
$$;

-- Row Level Security (matches project-wide pattern from migrations 001-007)
ALTER TABLE user_facts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_role_full_access" ON user_facts
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
