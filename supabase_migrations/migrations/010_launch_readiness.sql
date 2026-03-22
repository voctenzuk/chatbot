-- ============================================
-- Migration: 010_launch_readiness
-- Description: Launch-readiness fixes:
--   1. Add provider column to user_subscriptions (activate_subscription writes it)
--   2. Safety fix for NULL current_period_end on active subscriptions
--   3. Fix try_consume_photo expiry bug (secondary EXISTS check ignores period)
--   4. Add UNIQUE constraint on payments.provider_payment_id for idempotency
--   5. Add record_payment RPC for payment persistence
-- Depends on: 001_initial_schema, 008_character_mvp
-- ============================================

-- ============================================
-- 1. ADD PROVIDER COLUMN TO USER_SUBSCRIPTIONS
-- activate_subscription in db_client.py inserts "provider": "telegram_stars"
-- but the column didn't exist. Add it for analytics.
-- ============================================
ALTER TABLE user_subscriptions ADD COLUMN IF NOT EXISTS provider TEXT;

-- ============================================
-- 2. SAFETY: FIX NULL CURRENT_PERIOD_END
-- Active subscriptions with NULL period_end would bypass expiry checks.
-- Set to 30 days from period_start as a safe default.
-- ============================================
UPDATE user_subscriptions
SET current_period_end = current_period_start + INTERVAL '30 days'
WHERE current_period_end IS NULL AND status = 'active';

-- ============================================
-- 3. FIX TRY_CONSUME_PHOTO EXPIRY BUG
--
-- Bug: The original function (008_character_mvp.sql) has two queries:
--   1st: SELECT max_photos WHERE status='active' AND period valid → v_max_photos
--   2nd: EXISTS(... WHERE status='active') — NO period check
-- When a subscription expires but still has status='active', the 2nd query
-- returns TRUE, v_max_photos is NULL (not found in 1st query), and NULL
-- is treated as unlimited (Pro tier logic) → expired users get unlimited photos.
--
-- Fix: Single query for active non-expired subscription. If no active sub found,
-- fall back to Free plan limits (not hard-block) so expired Plus users still
-- get Free-tier photo access.
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
BEGIN
    -- Single query: get photo cap from active, non-expired subscription
    SELECT sp.max_photos_per_day INTO v_max_photos
    FROM user_subscriptions us
    JOIN subscription_plans sp ON us.plan_id = sp.id
    WHERE us.user_id = p_user_id
      AND us.status = 'active'
      AND us.current_period_start <= NOW()
      AND us.current_period_end >= NOW()
    ORDER BY us.created_at DESC
    LIMIT 1;

    -- No active, non-expired subscription → fallback to Free plan limits
    IF NOT FOUND THEN
        SELECT sp.max_photos_per_day INTO v_max_photos
        FROM subscription_plans sp
        WHERE sp.slug = 'free';

        -- Hardcoded fallback if Free plan row is missing
        v_max_photos := COALESCE(v_max_photos, 3);
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

    IF NOT FOUND THEN
        v_current_count := 0;
    END IF;

    -- At or over limit
    IF v_current_count >= v_max_photos THEN
        RETURN FALSE;
    END IF;

    -- Increment (note: narrow race window exists between SELECT and INSERT
    -- where two concurrent requests could both pass the limit check.
    -- Acceptable for MVP with 10 users. For scale, use INSERT...RETURNING
    -- with a single atomic check-and-increment.)
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
-- 4. UNIQUE CONSTRAINT ON PAYMENTS.PROVIDER_PAYMENT_ID
-- Enables idempotent INSERT ... ON CONFLICT DO NOTHING in record_payment.
-- ============================================
ALTER TABLE payments
    ADD CONSTRAINT payments_provider_payment_id_key UNIQUE (provider_payment_id);

-- ============================================
-- 5. RECORD_PAYMENT RPC
-- Persists payment records. Idempotent via ON CONFLICT DO NOTHING
-- on provider_payment_id to handle duplicate webhook/handler calls.
-- ============================================
CREATE OR REPLACE FUNCTION record_payment(
    p_user_id BIGINT,
    p_amount_cents INTEGER,
    p_provider_payment_id TEXT,
    p_status TEXT DEFAULT 'succeeded'
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO payments (user_id, amount_cents, status, provider_payment_id)
    VALUES (p_user_id, p_amount_cents, p_status, p_provider_payment_id)
    ON CONFLICT (provider_payment_id) DO NOTHING;

    -- Returns TRUE if a new row was inserted, FALSE if duplicate (ON CONFLICT)
    RETURN FOUND;
END;
$$;

GRANT EXECUTE ON FUNCTION record_payment(BIGINT, INTEGER, TEXT, TEXT) TO service_role;
