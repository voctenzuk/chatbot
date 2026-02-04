"""Tests for Subscription Service."""

import pytest

from bot.services.subscription_service import (
    SubscriptionService,
    SubscriptionTier,
    UserQuota,
    get_subscription_service,
    set_subscription_service,
)


class TestUserQuota:
    """Tests for UserQuota dataclass."""

    def test_free_user_limits(self):
        """Test free user has limits."""
        quota = UserQuota(user_id=123, tier=SubscriptionTier.FREE)

        assert quota.messages_remaining == 10
        assert quota.images_remaining == 3
        assert quota.can_send_message is True
        assert quota.can_generate_image is True

    def test_free_user_exhausted(self):
        """Test free user with exhausted limits."""
        quota = UserQuota(
            user_id=123,
            tier=SubscriptionTier.FREE,
            messages_used=10,
            images_used=3,
        )

        assert quota.messages_remaining == 0
        assert quota.images_remaining == 0
        assert quota.can_send_message is False
        assert quota.can_generate_image is False

    def test_premium_user_unlimited(self):
        """Test premium user has unlimited access."""
        quota = UserQuota(user_id=123, tier=SubscriptionTier.PREMIUM)

        assert quota.messages_remaining == float("inf")
        assert quota.images_remaining == float("inf")
        assert quota.can_send_message is True
        assert quota.can_generate_image is True


class TestSubscriptionService:
    """Tests for SubscriptionService."""

    @pytest.fixture
    def service(self):
        """Create fresh service instance."""
        return SubscriptionService()

    def test_get_quota_creates_new(self, service):
        """Test getting quota creates new entry."""
        quota = service.get_quota(123)

        assert isinstance(quota, UserQuota)
        assert quota.user_id == 123
        assert quota.tier == SubscriptionTier.FREE

    def test_use_message_success(self, service):
        """Test using message quota successfully."""
        success, quota = service.use_message(123)

        assert success is True
        assert quota.messages_used == 1

    def test_use_message_limit_reached(self, service):
        """Test using message when limit reached."""
        # Exhaust all messages
        for _ in range(10):
            service.use_message(123)

        success, quota = service.use_message(123)

        assert success is False
        assert quota.messages_used == 10

    def test_use_image_success(self, service):
        """Test using image quota successfully."""
        success, quota = service.use_image(123)

        assert success is True
        assert quota.images_used == 1

    def test_use_image_limit_reached(self, service):
        """Test using image when limit reached."""
        # Exhaust all images
        for _ in range(3):
            service.use_image(123)

        success, quota = service.use_image(123)

        assert success is False
        assert quota.images_used == 3

    def test_upgrade_to_premium(self, service):
        """Test upgrading user to premium."""
        service.use_message(123)  # Use some quota
        quota = service.upgrade_to_premium(123)

        assert quota.tier == SubscriptionTier.PREMIUM
        assert quota.can_send_message is True  # Unlimited now

    def test_get_usage_stats(self, service):
        """Test getting usage statistics."""
        service.use_message(123)
        service.use_image(123)

        stats = service.get_usage_stats(123)

        assert stats["tier"] == "free"
        assert stats["messages_used"] == 1
        assert stats["images_used"] == 1
        assert stats["messages_remaining"] == 9
        assert stats["images_remaining"] == 2

    def test_reset_quotas_single_user(self, service):
        """Test resetting quotas for single user."""
        service.use_message(123)
        service.use_image(123)

        service.reset_quotas(123)
        quota = service.get_quota(123)

        assert quota.messages_used == 0
        assert quota.images_used == 0

    def test_premium_not_counted_against_limit(self, service):
        """Test that premium users don't consume free quota."""
        service.upgrade_to_premium(123)

        for _ in range(20):
            success, _ = service.use_message(123)
            assert success is True

        # Should still have 0 used (doesn't count for premium)
        quota = service.get_quota(123)
        assert quota.messages_used == 0


class TestSubscriptionServiceGlobal:
    """Tests for global subscription service instance."""

    def test_get_subscription_service_creates_instance(self):
        """Test that get_subscription_service creates default instance."""
        set_subscription_service(None)  # Reset
        service = get_subscription_service()

        assert isinstance(service, SubscriptionService)

    def test_set_subscription_service(self):
        """Test setting global instance."""
        custom_service = SubscriptionService()
        set_subscription_service(custom_service)

        retrieved = get_subscription_service()
        assert retrieved is custom_service
