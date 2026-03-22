-- ============================================
-- Migration: 008_character_mvp
-- Description: Schema changes for Full Character MVP:
--   1. Fix cost_cents type (INTEGER → NUMERIC) for sub-cent precision
--   2. Add photo_count to usage_tracking
--   3. Add max_photos_per_day to subscription_plans
--   4. Atomic try_consume_photo RPC (race-safe)
--   5. Replace increment_usage with NUMERIC cost_cents
--   6. get_user_usage_today RPC for /stats
--   7. Modify provision_user_with_free_plan to return is_new
-- Depends on: 001_initial_schema, 004_monetization_updates
-- ============================================

-- ============================================
-- 1. FIX COST_CENTS TYPE
-- INTEGER truncates sub-cent costs ($0.001/msg = 0.1 cents → 0 as INTEGER)
-- NUMERIC(10,4) stores 0.0015 cents accurately
-- ============================================
ALTER TABLE usage_tracking ALTER COLUMN cost_cents TYPE NUMERIC(10,4);

-- ============================================
-- 2. ADD PHOTO_COUNT TO USAGE_TRACKING
-- Daily photo usage counter per user
-- ============================================
ALTER TABLE usage_tracking ADD COLUMN photo_count INTEGER DEFAULT 0;

-- ============================================
-- 3. ADD MAX_PHOTOS_PER_DAY TO SUBSCRIPTION_PLANS
-- Per-plan photo caps: Free=3, Plus=10, Pro=unlimited (NULL)
-- ============================================
ALTER TABLE subscription_plans ADD COLUMN max_photos_per_day INTEGER;
UPDATE subscription_plans SET max_photos_per_day = 3 WHERE slug = 'free';
UPDATE subscription_plans SET max_photos_per_day = 10 WHERE slug = 'plus';
UPDATE subscription_plans SET max_photos_per_day = NULL WHERE slug = 'pro';

-- ============================================
-- 4. ATOMIC TRY_CONSUME_PHOTO RPC
-- Check plan cap + increment photo_count in one transaction.
-- Returns TRUE if photo was consumed, FALSE if at limit or no subscription.
-- ============================================
CREATE OR REPLACE FUNCTION try_consume_photo(p_user_id BIGINT)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_max_photos INTEGER;
    v_current_count INTEGER;
    v_has_active_sub BOOLEAN;
BEGIN
    -- Get plan's photo cap from active subscription
    SELECT sp.max_photos_per_day INTO v_max_photos
    FROM user_subscriptions us
    JOIN subscription_plans sp ON us.plan_id = sp.id
    WHERE us.user_id = p_user_id
      AND us.status = 'active'
      AND us.current_period_start <= NOW()
      AND us.current_period_end >= NOW()
    ORDER BY us.created_at DESC
    LIMIT 1;

    -- Check if user has any active subscription at all
    SELECT EXISTS(
        SELECT 1 FROM user_subscriptions
        WHERE user_id = p_user_id AND status = 'active'
    ) INTO v_has_active_sub;

    -- No active subscription → blocked
    IF NOT v_has_active_sub THEN
        RETURN FALSE;
    END IF;

    -- NULL max_photos means unlimited (Pro tier)
    IF v_max_photos IS NULL THEN
        INSERT INTO usage_tracking (user_id, date, photo_count)
        VALUES (p_user_id, CURRENT_DATE, 1)
        ON CONFLICT (user_id, date)
        DO UPDATE SET photo_count = usage_tracking.photo_count + 1;
        RETURN TRUE;
    END IF;

    -- Check current count
    SELECT COALESCE(ut.photo_count, 0) INTO v_current_count
    FROM usage_tracking ut
    WHERE ut.user_id = p_user_id AND ut.date = CURRENT_DATE;

    -- No row yet means count is 0
    IF NOT FOUND THEN
        v_current_count := 0;
    END IF;

    -- At or over limit
    IF v_current_count >= v_max_photos THEN
        RETURN FALSE;
    END IF;

    -- Atomic increment
    INSERT INTO usage_tracking (user_id, date, photo_count)
    VALUES (p_user_id, CURRENT_DATE, 1)
    ON CONFLICT (user_id, date)
    DO UPDATE SET photo_count = usage_tracking.photo_count + 1;

    RETURN TRUE;
END;
$$;

GRANT EXECUTE ON FUNCTION try_consume_photo(BIGINT) TO authenticated;
GRANT EXECUTE ON FUNCTION try_consume_photo(BIGINT) TO service_role;

-- ============================================
-- 5. REPLACE INCREMENT_USAGE WITH NUMERIC COST_CENTS
-- Must drop old signature first (INTEGER → NUMERIC type change)
-- ============================================
DROP FUNCTION IF EXISTS increment_usage(BIGINT, INTEGER, INTEGER, INTEGER, INTEGER);

CREATE OR REPLACE FUNCTION increment_usage(
    p_user_id BIGINT,
    p_msg_count INTEGER DEFAULT 1,
    p_tokens_input INTEGER DEFAULT 0,
    p_tokens_output INTEGER DEFAULT 0,
    p_cost_cents NUMERIC(10,4) DEFAULT 0
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO usage_tracking (user_id, date, messages_sent, tokens_input, tokens_output, cost_cents)
    VALUES (p_user_id, CURRENT_DATE, p_msg_count, p_tokens_input, p_tokens_output, p_cost_cents)
    ON CONFLICT (user_id, date)
    DO UPDATE SET
        messages_sent = usage_tracking.messages_sent + EXCLUDED.messages_sent,
        tokens_input = usage_tracking.tokens_input + EXCLUDED.tokens_input,
        tokens_output = usage_tracking.tokens_output + EXCLUDED.tokens_output,
        cost_cents = usage_tracking.cost_cents + EXCLUDED.cost_cents;
END;
$$;

GRANT EXECUTE ON FUNCTION increment_usage(BIGINT, INTEGER, INTEGER, INTEGER, NUMERIC) TO authenticated;
GRANT EXECUTE ON FUNCTION increment_usage(BIGINT, INTEGER, INTEGER, INTEGER, NUMERIC) TO service_role;

-- ============================================
-- 6. GET_USER_USAGE_TODAY RPC FOR /stats
-- Returns today's usage, plan info, and days together
-- ============================================
CREATE OR REPLACE FUNCTION get_user_usage_today(p_user_id BIGINT)
RETURNS TABLE (
    messages_sent INTEGER,
    photo_count INTEGER,
    daily_limit INTEGER,
    photo_limit INTEGER,
    plan_slug TEXT,
    total_cost NUMERIC(10,4),
    days_together INTEGER
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT
        COALESCE(ut.messages_sent, 0)::INTEGER,
        COALESCE(ut.photo_count, 0)::INTEGER,
        sp.max_messages_per_day::INTEGER,
        sp.max_photos_per_day::INTEGER,
        sp.slug,
        COALESCE(ut.cost_cents, 0)::NUMERIC(10,4),
        COALESCE((CURRENT_DATE - u.created_at::date), 0)::INTEGER
    FROM users u
    LEFT JOIN user_subscriptions us
        ON u.telegram_id = us.user_id AND us.status = 'active'
    LEFT JOIN subscription_plans sp
        ON us.plan_id = sp.id
    LEFT JOIN usage_tracking ut
        ON u.telegram_id = ut.user_id AND ut.date = CURRENT_DATE
    WHERE u.telegram_id = p_user_id
    LIMIT 1;
END;
$$;

GRANT EXECUTE ON FUNCTION get_user_usage_today(BIGINT) TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_usage_today(BIGINT) TO service_role;

-- ============================================
-- 7. MODIFY PROVISION_USER_WITH_FREE_PLAN TO RETURN IS_NEW
-- Returns (user_id, is_new) instead of just BIGINT
-- ============================================
DROP FUNCTION IF EXISTS provision_user_with_free_plan(BIGINT, TEXT, TEXT);

CREATE OR REPLACE FUNCTION provision_user_with_free_plan(
    p_telegram_user_id BIGINT,
    p_username TEXT DEFAULT NULL,
    p_first_name TEXT DEFAULT NULL
)
RETURNS TABLE(user_id BIGINT, is_new BOOLEAN)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_is_new BOOLEAN := FALSE;
    v_free_plan_id UUID;
BEGIN
    -- Check if user exists before insert
    IF NOT EXISTS (SELECT 1 FROM users WHERE telegram_id = p_telegram_user_id) THEN
        v_is_new := TRUE;
    END IF;

    -- Upsert user
    INSERT INTO users (telegram_id, telegram_username, first_name)
    VALUES (p_telegram_user_id, p_username, p_first_name)
    ON CONFLICT (telegram_id) DO UPDATE SET
        telegram_username = COALESCE(EXCLUDED.telegram_username, users.telegram_username),
        first_name = COALESCE(EXCLUDED.first_name, users.first_name);

    -- If new user, assign Free plan
    IF v_is_new THEN
        SELECT id INTO v_free_plan_id
        FROM subscription_plans WHERE slug = 'free' LIMIT 1;

        IF v_free_plan_id IS NOT NULL THEN
            INSERT INTO user_subscriptions (
                user_id, plan_id, status,
                current_period_start, current_period_end
            ) VALUES (
                p_telegram_user_id, v_free_plan_id, 'active',
                NOW(), NOW() + INTERVAL '100 years'
            );
        END IF;
    END IF;

    RETURN QUERY SELECT p_telegram_user_id, v_is_new;
END;
$$;

GRANT EXECUTE ON FUNCTION provision_user_with_free_plan(BIGINT, TEXT, TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION provision_user_with_free_plan(BIGINT, TEXT, TEXT) TO service_role;
