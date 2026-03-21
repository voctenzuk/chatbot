"""Tests for Langfuse observability service."""

from unittest.mock import MagicMock, patch

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

    def test_initializes_with_settings(self):
        """Service creates Langfuse client from settings."""
        with (
            patch("bot.services.langfuse_service.settings") as mock_settings,
            patch("bot.services.langfuse_service.Langfuse") as mock_langfuse,
        ):
            mock_settings.langfuse_enabled = True
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            LangfuseService()

            mock_langfuse.assert_called_once_with(
                public_key="pk-test",
                secret_key="sk-test",
                host="https://cloud.langfuse.com",
            )


class TestCreateConfig:
    """Tests for config creation."""

    def test_returns_callbacks(self):
        """Returns config with callbacks list."""
        with (
            patch("bot.services.langfuse_service.settings") as mock_settings,
            patch("bot.services.langfuse_service.Langfuse"),
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

    def test_handler_receives_user_id_and_session(self):
        """CallbackHandler is created with user_id and session_id."""
        with (
            patch("bot.services.langfuse_service.settings") as mock_settings,
            patch("bot.services.langfuse_service.Langfuse"),
            patch("bot.services.langfuse_service.CallbackHandler") as mock_handler,
        ):
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://test.com"
            mock_settings.langfuse_enabled = True

            svc = LangfuseService()
            svc.create_config(user_id=42, session_id="ep_1", trace_name="chat")

            call_kwargs = mock_handler.call_args.kwargs
            assert call_kwargs["user_id"] == "42"
            assert call_kwargs["session_id"] == "ep_1"
            assert call_kwargs["trace_name"] == "chat"


class TestFlush:
    """Tests for flush on shutdown."""

    def test_flush_calls_client(self):
        """flush() calls client.flush() and shutdown()."""
        with (
            patch("bot.services.langfuse_service.settings") as mock_settings,
            patch("bot.services.langfuse_service.Langfuse") as mock_langfuse_cls,
        ):
            mock_settings.langfuse_enabled = True
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            mock_client = MagicMock()
            mock_langfuse_cls.return_value = mock_client

            svc = LangfuseService()
            svc.flush()

            mock_client.flush.assert_called_once()
            mock_client.shutdown.assert_called_once()


class TestSingleton:
    """Tests for DI get/set pattern."""

    def test_set_and_get(self):
        """set then get returns same instance."""
        with (
            patch("bot.services.langfuse_service.settings") as mock_settings,
            patch("bot.services.langfuse_service.Langfuse"),
        ):
            mock_settings.langfuse_enabled = True
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            custom = LangfuseService()
            set_langfuse_service(custom)
            assert get_langfuse_service() is custom
