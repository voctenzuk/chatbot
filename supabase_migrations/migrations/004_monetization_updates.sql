-- ============================================
-- Migration: 004_monetization_updates
-- Description: Add cost tracking to usage, Plus subscription tier,
--              and user provisioning with free plan auto-assignment
-- Depends on: 001_initial_schema
-- ============================================

-- ============================================
-- 1. ADD COST TRACKING TO USAGE
-- Tracks per-request cost in cents for billing analytics
-- ============================================
ALTER TABLE usage_tracking ADD COLUMN cost_cents INTEGER DEFAULT 0;

-- ============================================
-- 2. SEED DATA: PLUS PLAN
-- Mid-tier plan between Free and Pro
-- ============================================
INSERT INTO subscription_plans (name, slug, price_monthly_cents, max_messages_per_day, features)
VALUES ('Plus', 'plus', 499, 100,
  '{"gpt4": false, "priority_support": false, "advanced_features": false, "photos": true}'::jsonb);

-- ============================================
-- 3. REPLACE increment_usage FUNCTION
-- Adds p_cost_cents parameter to track per-request cost
-- ============================================
CREATE OR REPLACE FUNCTION increment_usage(
    p_user_id BIGINT,
    p_msg_count INTEGER DEFAULT 1,
    p_tokens_input INTEGER DEFAULT 0,
    p_tokens_output INTEGER DEFAULT 0,
    p_cost_cents INTEGER DEFAULT 0
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

GRANT EXECUTE ON FUNCTION increment_usage(BIGINT, INTEGER, INTEGER, INTEGER, INTEGER) TO authenticated;
GRANT EXECUTE ON FUNCTION increment_usage(BIGINT, INTEGER, INTEGER, INTEGER, INTEGER) TO service_role;

-- ============================================
-- 4. FUNCTION: provision_user_with_free_plan
-- Upserts a user and assigns the Free plan if no active subscription exists
-- ============================================
CREATE OR REPLACE FUNCTION provision_user_with_free_plan(
    p_telegram_user_id BIGINT,
    p_username TEXT DEFAULT NULL,
    p_first_name TEXT DEFAULT NULL
) RETURNS BIGINT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_free_plan_id UUID;
    v_has_subscription BOOLEAN;
BEGIN
    -- Upsert user
    INSERT INTO users (telegram_id, telegram_username, first_name)
    VALUES (p_telegram_user_id, p_username, p_first_name)
    ON CONFLICT (telegram_id) DO UPDATE SET
        telegram_username = COALESCE(EXCLUDED.telegram_username, users.telegram_username),
        first_name = COALESCE(EXCLUDED.first_name, users.first_name);

    -- Check if user already has an active subscription
    SELECT EXISTS(
        SELECT 1 FROM user_subscriptions
        WHERE user_id = p_telegram_user_id AND status = 'active'
    ) INTO v_has_subscription;

    -- If no active subscription, assign Free plan
    IF NOT v_has_subscription THEN
        SELECT id INTO v_free_plan_id FROM subscription_plans WHERE slug = 'free' LIMIT 1;

        IF v_free_plan_id IS NOT NULL THEN
            INSERT INTO user_subscriptions (user_id, plan_id, status, current_period_start, current_period_end)
            VALUES (
                p_telegram_user_id,
                v_free_plan_id,
                'active',
                NOW(),
                NOW() + INTERVAL '100 years'  -- Free plan never expires
            );
        END IF;
    END IF;

    RETURN p_telegram_user_id;
END;
$$;

GRANT EXECUTE ON FUNCTION provision_user_with_free_plan(BIGINT, TEXT, TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION provision_user_with_free_plan(BIGINT, TEXT, TEXT) TO service_role;
