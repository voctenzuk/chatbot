"""Tests for monetization features: rate limiting, usage tracking, cost estimation, payments."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aiogram.types import Chat, User

from bot.services.db_client import DatabaseClient
from bot.services.llm_service import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockMessage:
    """Minimal Telegram message mock for handler tests."""

    text: str | None
    caption: str | None
    from_user: Any
    chat: Any
    photo: Any
    document: Any
    voice: Any
    video: Any
    audio: Any
    sticker: Any
    location: Any
    contact: Any

    def __init__(
        self,
        text: str | None = None,
        caption: str | None = None,
        user_id: int = 12345,
        first_name: str | None = None,
    ):
        self.text = text
        self.caption = caption
        self.from_user = MagicMock(spec=User)
        self.from_user.id = user_id
        self.from_user.first_name = first_name
        self.from_user.username = "testuser"
        self.chat = MagicMock(spec=Chat)
        self.chat.id = user_id
        self.photo = None
        self.document = None
        self.voice = None
        self.video = None
        self.audio = None
        self.sticker = None
        self.location = None
        self.contact = None
        self.successful_payment = None
        self._last_answer: str | None = None
        self._last_invoice: dict[str, Any] | None = None

    async def answer(self, text: str, **kwargs: Any) -> MagicMock:
        self._last_answer = text
        return MagicMock()

    async def answer_invoice(self, **kwargs: Any) -> MagicMock:
        self._last_invoice = kwargs
        return MagicMock()


def _make_message_result(episode_id: str | None = None) -> MagicMock:
    """Build a mock MessageResult for episode_manager."""
    mock_episode = MagicMock()
    mock_episode.id = episode_id or str(uuid4())

    mock_msg = MagicMock()
    mock_msg.id = str(uuid4())
    mock_msg.episode_id = mock_episode.id

    mock_decision = MagicMock()
    mock_decision.should_switch = False
    mock_decision.reason = "Continuing"
    mock_decision.confidence = 0.5
    mock_decision.trigger_type = None

    result = MagicMock()
    result.message = mock_msg
    result.episode = mock_episode
    result.is_new_episode = False
    result.switch_decision = mock_decision
    return result


def _make_mock_rpc(return_data: Any = True) -> MagicMock:
    """Build a mock Supabase client with rpc().execute() chain."""
    mock_response = MagicMock()
    mock_response.data = return_data

    mock_execute = MagicMock()
    mock_execute.execute = MagicMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.rpc = MagicMock(return_value=mock_execute)
    return mock_client


# ---------------------------------------------------------------------------
# TestDbClientRateLimit
# ---------------------------------------------------------------------------


class TestDbClientRateLimit:
    """Tests for db_client.check_rate_limit method."""

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self) -> None:
        """RPC returns True -> method returns True."""
        mock_client = _make_mock_rpc(return_data=True)
        db = DatabaseClient(client=mock_client)

        result = await db.check_rate_limit(telegram_user_id=42)

        assert result is True
        mock_client.rpc.assert_called_once_with(
            "check_rate_limit",
            {"p_user_id": 42},
        )

    @pytest.mark.asyncio
    async def test_check_rate_limit_blocked(self) -> None:
        """RPC returns False -> method returns False."""
        mock_client = _make_mock_rpc(return_data=False)
        db = DatabaseClient(client=mock_client)

        result = await db.check_rate_limit(telegram_user_id=42)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_error_propagates(self) -> None:
        """RPC raises -> exception propagates to caller."""
        mock_client = MagicMock()
        mock_execute = MagicMock()
        mock_execute.execute = MagicMock(side_effect=RuntimeError("db down"))
        mock_client.rpc = MagicMock(return_value=mock_execute)
        db = DatabaseClient(client=mock_client)

        with pytest.raises(RuntimeError, match="db down"):
            await db.check_rate_limit(telegram_user_id=42)


# ---------------------------------------------------------------------------
# TestDbClientUsageTracking
# ---------------------------------------------------------------------------


class TestDbClientUsageTracking:
    """Tests for db_client.increment_usage method."""

    @pytest.mark.asyncio
    async def test_increment_usage_success(self) -> None:
        """RPC called with correct params including cost_cents."""
        mock_client = _make_mock_rpc()
        db = DatabaseClient(client=mock_client)

        await db.increment_usage(
            telegram_user_id=42,
            msg_count=1,
            tokens_in=100,
            tokens_out=50,
            cost_cents=2,
        )

        mock_client.rpc.assert_called_once_with(
            "increment_usage",
            {
                "p_user_id": 42,
                "p_msg_count": 1,
                "p_tokens_input": 100,
                "p_tokens_output": 50,
                "p_cost_cents": 2,
            },
        )

    @pytest.mark.asyncio
    async def test_increment_usage_with_cost(self) -> None:
        """cost_cents parameter passed through to RPC."""
        mock_client = _make_mock_rpc()
        db = DatabaseClient(client=mock_client)

        await db.increment_usage(telegram_user_id=7, cost_cents=99)

        call_args = mock_client.rpc.call_args
        assert call_args[0][1]["p_cost_cents"] == 99

    @pytest.mark.asyncio
    async def test_increment_usage_error_swallowed(self) -> None:
        """RPC raises -> warning logged, no exception raised."""
        mock_client = MagicMock()
        mock_execute = MagicMock()
        mock_execute.execute = MagicMock(side_effect=RuntimeError("db write failed"))
        mock_client.rpc = MagicMock(return_value=mock_execute)
        db = DatabaseClient(client=mock_client)

        # Should NOT raise
        await db.increment_usage(telegram_user_id=42, msg_count=1)


# ---------------------------------------------------------------------------
# TestDbClientProvisioning
# ---------------------------------------------------------------------------


class TestDbClientProvisioning:
    """Tests for db_client.provision_user method."""

    @pytest.mark.asyncio
    async def test_provision_new_user(self) -> None:
        """RPC called with telegram_user_id, username, first_name."""
        mock_client = _make_mock_rpc()
        db = DatabaseClient(client=mock_client)

        await db.provision_user(
            telegram_user_id=42,
            username="alice",
            first_name="Alice",
        )

        mock_client.rpc.assert_called_once_with(
            "provision_user_with_free_plan",
            {
                "p_telegram_user_id": 42,
                "p_username": "alice",
                "p_first_name": "Alice",
            },
        )

    @pytest.mark.asyncio
    async def test_provision_user_error_swallowed(self) -> None:
        """RPC raises -> warning logged, no exception raised."""
        mock_client = MagicMock()
        mock_execute = MagicMock()
        mock_execute.execute = MagicMock(side_effect=RuntimeError("provision failed"))
        mock_client.rpc = MagicMock(return_value=mock_execute)
        db = DatabaseClient(client=mock_client)

        # Should NOT raise
        await db.provision_user(telegram_user_id=42, username="bob", first_name="Bob")


# ---------------------------------------------------------------------------
# TestCostCalculation
# ---------------------------------------------------------------------------


class TestCostCalculation:
    """Tests for estimate_cost_cents function."""

    def test_known_model_cost(self) -> None:
        """kimi-k2p5 uses specific rates."""
        from bot.services.llm_service import estimate_cost_cents

        # 10000 * 0.1 / 1000 + 5000 * 0.3 / 1000 = 1.0 + 1.5 = 2.5 -> int(2.5) = 2
        result = estimate_cost_cents("kimi-k2p5", 10000, 5000)
        assert result == 2

    def test_unknown_model_uses_default(self) -> None:
        """Unknown model falls back to default rates."""
        from bot.services.llm_service import estimate_cost_cents

        # Same default rates -> same result
        result = estimate_cost_cents("unknown-model", 10000, 5000)
        assert result == 2

    def test_zero_tokens_returns_zero(self) -> None:
        """Zero tokens -> zero cost."""
        from bot.services.llm_service import estimate_cost_cents

        assert estimate_cost_cents("kimi-k2p5", 0, 0) == 0


# ---------------------------------------------------------------------------
# TestHandlerMonetization — fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_episode_manager() -> AsyncMock:
    mgr = AsyncMock()
    mgr.process_user_message = AsyncMock(return_value=_make_message_result())
    mgr.process_assistant_message = AsyncMock(return_value=_make_message_result())
    mgr.get_recent_messages = AsyncMock(return_value=[])
    return mgr


@pytest.fixture
def mock_memory_service() -> AsyncMock:
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    return svc


@pytest.fixture
def mock_llm_service() -> AsyncMock:
    svc = AsyncMock()
    svc.generate = AsyncMock(
        return_value=LLMResponse(
            content="LLM reply text",
            model="test-model",
            tokens_in=15,
            tokens_out=8,
        )
    )
    return svc


@pytest.fixture
def mock_context_builder() -> MagicMock:
    builder = MagicMock()
    builder.assemble_for_llm = MagicMock(
        return_value=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ]
    )
    return builder


@pytest.fixture
def mock_db_client() -> AsyncMock:
    db = AsyncMock()
    db.check_rate_limit = AsyncMock(return_value=True)
    db.increment_usage = AsyncMock()
    db.provision_user = AsyncMock()
    db.activate_subscription = AsyncMock()
    return db


@pytest.fixture
def patched_handlers(
    mock_episode_manager: AsyncMock,
    mock_memory_service: AsyncMock,
    mock_llm_service: AsyncMock,
    mock_context_builder: MagicMock,
    mock_db_client: AsyncMock,
) -> Any:
    """Patch all service accessors used in handlers."""
    with (
        patch(
            "bot.handlers.get_episode_manager_service",
            return_value=mock_episode_manager,
        ),
        patch(
            "bot.handlers.get_memory_service",
            return_value=mock_memory_service,
        ),
        patch(
            "bot.handlers.get_llm_service",
            return_value=mock_llm_service,
        ),
        patch(
            "bot.handlers.get_context_builder",
            return_value=mock_context_builder,
        ),
        patch(
            "bot.handlers.get_system_prompt",
            return_value="You are a helpful assistant.",
        ),
        patch(
            "bot.handlers.DB_CLIENT_AVAILABLE",
            True,
        ),
        patch(
            "bot.handlers.get_db_client",
            return_value=mock_db_client,
        ),
    ):
        yield {
            "episode_manager": mock_episode_manager,
            "memory_service": mock_memory_service,
            "llm_service": mock_llm_service,
            "context_builder": mock_context_builder,
            "db_client": mock_db_client,
        }


# ---------------------------------------------------------------------------
# TestHandlerMonetization
# ---------------------------------------------------------------------------


class TestHandlerMonetization:
    """Tests for monetization wiring in handlers."""

    @pytest.mark.asyncio
    async def test_chat_rate_limit_pass_continues(self, patched_handlers: dict[str, Any]) -> None:
        """When rate limit check returns True, LLM is called normally."""
        from bot.handlers import chat

        patched_handlers["db_client"].check_rate_limit = AsyncMock(return_value=True)

        msg = MockMessage(text="Hello!", user_id=42)
        await chat(msg)

        patched_handlers["llm_service"].generate.assert_called_once()
        assert msg._last_answer == "LLM reply text"

    @pytest.mark.asyncio
    async def test_chat_rate_limit_blocked(self, patched_handlers: dict[str, Any]) -> None:
        """When rate limit check returns False, limit message shown, no LLM call."""
        from bot.handlers import chat

        patched_handlers["db_client"].check_rate_limit = AsyncMock(return_value=False)

        msg = MockMessage(text="Hello!", user_id=42)
        await chat(msg)

        assert msg._last_answer is not None
        assert "лимит" in msg._last_answer
        patched_handlers["llm_service"].generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_rate_limit_error_fails_open(self, patched_handlers: dict[str, Any]) -> None:
        """When rate limit check raises, message proceeds (fail open)."""
        from bot.handlers import chat

        patched_handlers["db_client"].check_rate_limit = AsyncMock(
            side_effect=RuntimeError("db down")
        )

        msg = MockMessage(text="Hello!", user_id=42)
        await chat(msg)

        # LLM was still called despite rate limit error (fail open)
        patched_handlers["llm_service"].generate.assert_called_once()
        assert msg._last_answer == "LLM reply text"

    @pytest.mark.asyncio
    async def test_chat_increments_usage_after_response(
        self, patched_handlers: dict[str, Any]
    ) -> None:
        """After LLM response, increment_usage is called with token counts."""
        from bot.handlers import chat

        patched_handlers["db_client"].check_rate_limit = AsyncMock(return_value=True)

        msg = MockMessage(text="Hello!", user_id=42)
        await chat(msg)

        patched_handlers["db_client"].increment_usage.assert_called_once()
        call_kwargs = patched_handlers["db_client"].increment_usage.call_args
        # Verify user_id is the first positional arg
        assert call_kwargs[0][0] == 42
        # Verify token counts are passed
        assert call_kwargs[1]["tokens_in"] == 15
        assert call_kwargs[1]["tokens_out"] == 8

    @pytest.mark.asyncio
    async def test_upgrade_sends_invoice(self, patched_handlers: dict[str, Any]) -> None:
        """The /upgrade command sends a Telegram Stars invoice."""
        try:
            from bot.handlers import upgrade
        except ImportError:
            pytest.skip("upgrade handler not yet implemented")

        msg = MockMessage(text="/upgrade", user_id=42)
        await upgrade(msg)

        assert msg._last_invoice is not None
        assert msg._last_invoice.get("currency") == "XTR"

    @pytest.mark.asyncio
    async def test_pre_checkout_valid_plan(self, patched_handlers: dict[str, Any]) -> None:
        """Valid plan payload -> answer(ok=True)."""
        try:
            from bot.handlers import pre_checkout
        except ImportError:
            pytest.skip("pre_checkout handler not yet implemented")

        query = AsyncMock()
        query.invoice_payload = "plan:plus"
        query.from_user = MagicMock(spec=User)
        query.from_user.id = 42
        query.answer = AsyncMock()

        await pre_checkout(query)

        query.answer.assert_called_once()
        call_kwargs = query.answer.call_args
        assert call_kwargs[1].get("ok") is True or (call_kwargs[0] and call_kwargs[0][0] is True)

    @pytest.mark.asyncio
    async def test_pre_checkout_invalid(self, patched_handlers: dict[str, Any]) -> None:
        """Invalid payload -> answer(ok=False)."""
        try:
            from bot.handlers import pre_checkout
        except ImportError:
            pytest.skip("pre_checkout handler not yet implemented")

        query = AsyncMock()
        query.invoice_payload = "garbage"
        query.from_user = MagicMock(spec=User)
        query.from_user.id = 42
        query.answer = AsyncMock()

        await pre_checkout(query)

        query.answer.assert_called_once()
        call_kwargs = query.answer.call_args
        assert call_kwargs[1].get("ok") is False or (call_kwargs[0] and call_kwargs[0][0] is False)

    @pytest.mark.asyncio
    async def test_successful_payment_activates(self, patched_handlers: dict[str, Any]) -> None:
        """Successful payment -> activate_subscription called."""
        try:
            from bot.handlers import successful_payment
        except ImportError:
            pytest.skip("successful_payment handler not yet implemented")

        mock_payment = MagicMock()
        mock_payment.invoice_payload = "plan:plus"
        mock_payment.currency = "XTR"
        mock_payment.total_amount = 100

        msg = MockMessage(text=None, user_id=42)
        msg.successful_payment = mock_payment
        await successful_payment(msg)

        patched_handlers["db_client"].activate_subscription.assert_called_once()
        call_args = patched_handlers["db_client"].activate_subscription.call_args
        assert call_args[0][0] == 42  # telegram_user_id
        assert "plus" in call_args[0][1]  # plan_slug contains "plus"
