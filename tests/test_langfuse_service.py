"""Tests for Langfuse observability service."""

from unittest.mock import MagicMock, patch

import pytest

from bot.infra.langfuse_service import (
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
            patch("bot.infra.langfuse_service.settings") as mock_settings,
            patch("bot.infra.langfuse_service.Langfuse") as mock_langfuse,
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


class TestTrace:
    """Tests for trace() context manager."""

    def test_trace_calls_propagate_attributes(self):
        """trace() wraps propagate_attributes with correct args."""
        with (
            patch("bot.infra.langfuse_service.settings") as mock_settings,
            patch("bot.infra.langfuse_service.Langfuse"),
            patch("bot.infra.langfuse_service.propagate_attributes") as mock_prop,
        ):
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            svc = LangfuseService()
            with svc.trace(
                user_id=123,
                session_id="ep_456",
                trace_name="chat",
                tags=["test"],
            ):
                pass

            mock_prop.assert_called_once_with(
                user_id="123",
                session_id="ep_456",
                trace_name="chat",
                tags=["test"],
            )

    def test_trace_converts_user_id_to_string(self):
        """user_id int is converted to string for propagate_attributes."""
        with (
            patch("bot.infra.langfuse_service.settings") as mock_settings,
            patch("bot.infra.langfuse_service.Langfuse"),
            patch("bot.infra.langfuse_service.propagate_attributes") as mock_prop,
        ):
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            svc = LangfuseService()
            with svc.trace(user_id=42):
                pass

            call_kwargs = mock_prop.call_args.kwargs
            assert call_kwargs["user_id"] == "42"
            assert isinstance(call_kwargs["user_id"], str)

    def test_trace_defaults(self):
        """trace() uses default trace_name and empty tags."""
        with (
            patch("bot.infra.langfuse_service.settings") as mock_settings,
            patch("bot.infra.langfuse_service.Langfuse"),
            patch("bot.infra.langfuse_service.propagate_attributes") as mock_prop,
        ):
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            svc = LangfuseService()
            with svc.trace(user_id=1):
                pass

            mock_prop.assert_called_once_with(
                user_id="1",
                session_id=None,
                trace_name="chat",
                tags=[],
            )


class TestFlush:
    """Tests for flush on shutdown."""

    def test_flush_calls_client(self):
        """flush() calls client.flush() and shutdown()."""
        with (
            patch("bot.infra.langfuse_service.settings") as mock_settings,
            patch("bot.infra.langfuse_service.Langfuse") as mock_langfuse_cls,
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
            patch("bot.infra.langfuse_service.settings") as mock_settings,
            patch("bot.infra.langfuse_service.Langfuse"),
        ):
            mock_settings.langfuse_enabled = True
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_base_url = "https://cloud.langfuse.com"

            custom = LangfuseService()
            set_langfuse_service(custom)
            assert get_langfuse_service() is custom
