"""Tests for port protocol definitions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.ports import LLMPort, MemoryPort, MessageDeliveryPort


class TestLLMPort:
    """Verify LLMPort is structurally compatible with LLMService."""

    @pytest.mark.asyncio
    async def test_llm_service_satisfies_protocol(self) -> None:
        from bot.services.llm_service import LLMService

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="hi",
                usage_metadata={"input_tokens": 1, "output_tokens": 1},
                response_metadata={"model_name": "test"},
                tool_calls=[],
            )
        )
        svc = LLMService(model=mock_model)
        assert isinstance(svc, LLMPort)

    @pytest.mark.asyncio
    async def test_mock_satisfies_protocol(self) -> None:
        mock = AsyncMock(spec=LLMPort)
        assert isinstance(mock, LLMPort)


class TestMemoryPort:
    """Verify MemoryPort is structurally compatible with CogneeMemoryService."""

    @pytest.mark.asyncio
    async def test_mock_satisfies_protocol(self) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(return_value=[])
        mock.write_factual = AsyncMock(return_value="mem-id")
        mock.cognify = AsyncMock()
        assert isinstance(mock, MemoryPort)


class TestMessageDeliveryPort:
    """Verify MessageDeliveryPort protocol."""

    @pytest.mark.asyncio
    async def test_mock_satisfies_protocol(self) -> None:
        mock = MagicMock()
        mock.send_text = AsyncMock()
        mock.send_photo = AsyncMock()
        assert isinstance(mock, MessageDeliveryPort)
