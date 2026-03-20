-- ============================================
-- Migration: 001_initial_schema
-- Description: Initial schema for Telegram AI Companion Bot
-- ============================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================
-- 1. USERS TABLE
-- Maps Telegram users to application users
-- ============================================
CREATE TABLE users (
    telegram_id BIGINT PRIMARY KEY,
    telegram_username TEXT,
    first_name TEXT,
    last_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for faster lookups
CREATE INDEX idx_users_created_at ON users(created_at);

-- ============================================
-- 2. SUBSCRIPTION PLANS
-- Available subscription tiers
-- ============================================
CREATE TABLE subscription_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    price_monthly_cents INTEGER,
    max_messages_per_day INTEGER DEFAULT NULL,
    features JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- 3. USER SUBSCRIPTIONS
-- Active and historical subscriptions
-- ============================================
CREATE TABLE user_subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    plan_id UUID NOT NULL REFERENCES subscription_plans(id),
    status TEXT NOT NULL CHECK (status IN ('active', 'canceled', 'past_due', 'unpaid', 'trialing', 'paused')),
    current_period_start TIMESTAMPTZ NOT NULL,
    current_period_end TIMESTAMPTZ NOT NULL,
    provider_subscription_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for subscriptions
CREATE INDEX idx_subscriptions_user_id ON user_subscriptions(user_id);
CREATE INDEX idx_subscriptions_status ON user_subscriptions(status);
CREATE INDEX idx_subscriptions_period_end ON user_subscriptions(current_period_end);

-- ============================================
-- 4. USAGE TRACKING
-- Daily usage counters per user
-- ============================================
CREATE TABLE usage_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    messages_sent INTEGER DEFAULT 0,
    tokens_input INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- One record per user per day
    CONSTRAINT unique_user_daily_usage UNIQUE (user_id, date)
);

-- Indexes for usage tracking
CREATE INDEX idx_usage_user_date ON usage_tracking(user_id, date);
CREATE INDEX idx_usage_date ON usage_tracking(date);

-- ============================================
-- 5. PAYMENTS
-- Payment history
-- ============================================
CREATE TABLE payments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE,
    amount_cents INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'succeeded', 'failed', 'refunded', 'disputed')),
    provider_payment_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for payments
CREATE INDEX idx_payments_user_id ON payments(user_id);
CREATE INDEX idx_payments_status ON payments(status);
CREATE INDEX idx_payments_created ON payments(created_at);

-- ============================================
-- 6. WEBHOOK EVENTS
-- For processing async payment webhooks
-- ============================================
CREATE TABLE webhook_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for webhook events
CREATE INDEX idx_webhooks_status ON webhook_events(status);
CREATE INDEX idx_webhooks_provider ON webhook_events(provider, event_type);
CREATE INDEX idx_webhooks_created ON webhook_events(created_at);

-- ============================================
-- ENABLE ROW LEVEL SECURITY ON ALL TABLES
-- ============================================
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscription_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE payments ENABLE ROW LEVEL SECURITY;
ALTER TABLE webhook_events ENABLE ROW LEVEL SECURITY;

-- ============================================
-- RLS POLICIES
-- ============================================

-- --------------------------------------------
-- SERVICE ROLE POLICIES (Full access)
-- --------------------------------------------

CREATE POLICY "Service role full access on users"
ON users FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access on subscription_plans"
ON subscription_plans FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access on user_subscriptions"
ON user_subscriptions FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access on usage_tracking"
ON usage_tracking FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access on payments"
ON payments FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

CREATE POLICY "Service role full access on webhook_events"
ON webhook_events FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- --------------------------------------------
-- USER POLICIES (Access own data only)
-- Filter by telegram_id from JWT claim
-- --------------------------------------------

-- Users can read their own profile
CREATE POLICY "Users can read own profile"
ON users FOR SELECT
TO authenticated
USING (telegram_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can update their own profile (limited fields)
CREATE POLICY "Users can update own profile"
ON users FOR UPDATE
TO authenticated
USING (telegram_id = ((auth.jwt() ->> 'telegram_id')::BIGINT))
WITH CHECK (telegram_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can read their own subscriptions
CREATE POLICY "Users can read own subscriptions"
ON user_subscriptions FOR SELECT
TO authenticated
USING (user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can read their own usage
CREATE POLICY "Users can read own usage"
ON usage_tracking FOR SELECT
TO authenticated
USING (user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- Users can read their own payments
CREATE POLICY "Users can read own payments"
ON payments FOR SELECT
TO authenticated
USING (user_id = ((auth.jwt() ->> 'telegram_id')::BIGINT));

-- --------------------------------------------
-- PUBLIC POLICIES
-- --------------------------------------------

-- Public can read subscription plans (for pricing page)
CREATE POLICY "Public can read subscription plans"
ON subscription_plans FOR SELECT
TO anon, authenticated
USING (true);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- --------------------------------------------
-- Function: check_rate_limit
-- Checks if user has exceeded their daily message limit
-- Returns: boolean (true if within limit, false if exceeded)
-- --------------------------------------------
CREATE OR REPLACE FUNCTION check_rate_limit(p_user_id BIGINT)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_max_messages INTEGER;
    v_messages_sent INTEGER;
    v_plan_slug TEXT;
BEGIN
    -- Get the user's active plan and daily limit
    SELECT 
        sp.max_messages_per_day,
        sp.slug
    INTO v_max_messages, v_plan_slug
    FROM user_subscriptions us
    JOIN subscription_plans sp ON us.plan_id = sp.id
    WHERE us.user_id = p_user_id
      AND us.status = 'active'
      AND us.current_period_start <= NOW()
      AND us.current_period_end >= NOW()
    ORDER BY us.created_at DESC
    LIMIT 1;
    
    -- If no active subscription found, default to 0 (blocked)
    IF v_plan_slug IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- If max_messages_per_day is NULL, user has unlimited messages
    IF v_max_messages IS NULL THEN
        RETURN TRUE;
    END IF;
    
    -- Get today's message count
    SELECT COALESCE(messages_sent, 0)
    INTO v_messages_sent
    FROM usage_tracking
    WHERE user_id = p_user_id
      AND date = CURRENT_DATE;
    
    -- Return true if within limit, false otherwise
    RETURN COALESCE(v_messages_sent, 0) < v_max_messages;
END;
$$;

-- --------------------------------------------
-- Function: increment_usage
-- Increments the usage counters for a user
-- Parameters:
--   p_user_id: The user's telegram_id
--   p_msg_count: Number of messages to increment (typically 1)
--   p_tokens_input: Input tokens used
--   p_tokens_output: Output tokens used
-- --------------------------------------------
CREATE OR REPLACE FUNCTION increment_usage(
    p_user_id BIGINT,
    p_msg_count INTEGER DEFAULT 1,
    p_tokens_input INTEGER DEFAULT 0,
    p_tokens_output INTEGER DEFAULT 0
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO usage_tracking (user_id, date, messages_sent, tokens_input, tokens_output)
    VALUES (p_user_id, CURRENT_DATE, p_msg_count, p_tokens_input, p_tokens_output)
    ON CONFLICT (user_id, date)
    DO UPDATE SET 
        messages_sent = usage_tracking.messages_sent + EXCLUDED.messages_sent,
        tokens_input = usage_tracking.tokens_input + EXCLUDED.tokens_input,
        tokens_output = usage_tracking.tokens_output + EXCLUDED.tokens_output;
END;
$$;

-- --------------------------------------------
-- Function: get_user_usage_today
-- Helper to get today's usage for a user
-- Returns: record with messages_sent, tokens_input, tokens_output
-- --------------------------------------------
CREATE OR REPLACE FUNCTION get_user_usage_today(p_user_id BIGINT)
RETURNS TABLE (
    messages_sent INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    daily_limit INTEGER
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(ut.messages_sent, 0),
        COALESCE(ut.tokens_input, 0),
        COALESCE(ut.tokens_output, 0),
        sp.max_messages_per_day
    FROM users u
    LEFT JOIN usage_tracking ut ON u.telegram_id = ut.user_id AND ut.date = CURRENT_DATE
    LEFT JOIN user_subscriptions us ON u.telegram_id = us.user_id AND us.status = 'active'
    LEFT JOIN subscription_plans sp ON us.plan_id = sp.id
    WHERE u.telegram_id = p_user_id
    LIMIT 1;
END;
$$;

-- ============================================
-- SEED DATA
-- ============================================

-- Insert Free plan
INSERT INTO subscription_plans (name, slug, price_monthly_cents, max_messages_per_day, features)
VALUES (
    'Free',
    'free',
    0,
    20,
    '{"gpt4": false, "priority_support": false, "advanced_features": false}'::jsonb
);

-- Insert Pro plan
INSERT INTO subscription_plans (name, slug, price_monthly_cents, max_messages_per_day, features)
VALUES (
    'Pro',
    'pro',
    999,
    NULL,  -- Unlimited messages
    '{"gpt4": true, "priority_support": true, "advanced_features": true}'::jsonb
);

-- ============================================
-- GRANT PERMISSIONS
-- ============================================

-- Grant usage on functions to authenticated users
GRANT EXECUTE ON FUNCTION check_rate_limit(BIGINT) TO authenticated;
GRANT EXECUTE ON FUNCTION increment_usage(BIGINT, INTEGER, INTEGER, INTEGER) TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_usage_today(BIGINT) TO authenticated;

-- Grant usage on functions to service_role
GRANT EXECUTE ON FUNCTION check_rate_limit(BIGINT) TO service_role;
GRANT EXECUTE ON FUNCTION increment_usage(BIGINT, INTEGER, INTEGER, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION get_user_usage_today(BIGINT) TO service_role;
