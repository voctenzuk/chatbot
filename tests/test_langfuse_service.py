"""Tests for Langfuse observability service."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bot.services.langfuse_service import (
    LangfuseService,
    get_langfuse_service,
    set_langfuse_service,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    set_langfuse_service(None)
    yield
    set_langfuse_service(None)


class TestLangfuseServiceInit:
    """Tests for LangfuseService initialization."""

    def test_unavailable_without_keys(self):
        """Service is unavailable when keys not configured."""
        with patch("bot.services.langfuse_service.settings") as mock_settings:
            mock_settings.langfuse_enabled = True
            mock_settings.langfuse_public_key = None
            mock_settings.langfuse_secret_key = None
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            svc = LangfuseService()
            assert not svc.available

    def test_unavailable_when_disabled(self):
        """Service is unavailable when disabled via config."""
        with patch("bot.services.langfuse_service.settings") as mock_settings:
            mock_settings.langfuse_enabled = False
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            svc = LangfuseService()
            assert not svc.available

    def test_available_with_keys(self):
        """Service is available when keys are configured."""
        with (
            patch("bot.services.langfuse_service.LANGFUSE_AVAILABLE", True),
            patch("bot.services.langfuse_service.settings") as mock_settings,
        ):
            mock_settings.langfuse_enabled = True
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            svc = LangfuseService()
            assert svc.available


class TestCreateConfig:
    """Tests for config creation."""

    def test_returns_empty_when_unavailable(self):
        """Returns empty dict when service is unavailable."""
        with patch("bot.services.langfuse_service.settings") as mock_settings:
            mock_settings.langfuse_enabled = True
            mock_settings.langfuse_public_key = None
            mock_settings.langfuse_secret_key = None
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            svc = LangfuseService()
            config = svc.create_config(user_id=123)
            assert config == {}

    def test_returns_callbacks_when_available(self):
        """Returns config with callbacks when service is available."""
        with (
            patch("bot.services.langfuse_service.LANGFUSE_AVAILABLE", True),
            patch("bot.services.langfuse_service.settings") as mock_settings,
            patch("bot.services.langfuse_service.CallbackHandler") as mock_handler,
        ):
            mock_settings.langfuse_enabled = True
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            svc = LangfuseService()
            config = svc.create_config(
                user_id=123,
                session_id="ep_456",
                trace_name="chat",
                tags=["test"],
            )
            assert "callbacks" in config
            assert len(config["callbacks"]) == 1
            mock_handler.assert_called_once()


class TestSingleton:
    """Tests for DI get/set pattern."""

    def test_get_returns_instance(self):
        """get_langfuse_service returns an instance."""
        svc = get_langfuse_service()
        assert isinstance(svc, LangfuseService)

    def test_set_and_get(self):
        """set then get returns same instance."""
        with patch("bot.services.langfuse_service.settings") as mock_settings:
            mock_settings.langfuse_enabled = False
            mock_settings.langfuse_public_key = None
            mock_settings.langfuse_secret_key = None
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            custom = LangfuseService()
            set_langfuse_service(custom)
            assert get_langfuse_service() is custom
