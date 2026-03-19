"""Tests for LLM service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.services.llm_service import (
    LLMResponse,
    LLMService,
    get_llm_service,
    set_llm_service,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset LLM service singleton between tests."""
    set_llm_service(None)
    yield
    set_llm_service(None)


class TestLLMServiceInit:
    """Tests for LLMService initialization."""

    def test_llm_service_init_defaults_from_settings(self):
        """LLMService() reads model/url/key from settings when no explicit args."""
        with patch("bot.services.llm_service.settings") as mock_settings:
            mock_settings.llm_model = "test-model"
            mock_settings.llm_base_url = "https://test.api"
            mock_settings.llm_api_key = "sk-test"
            mock_settings.llm_temperature = 0.5
            mock_settings.llm_max_tokens = 512

            with patch("bot.services.llm_service.ChatOpenAI") as mock_chat:
                _svc = LLMService()  # noqa: F841

            mock_chat.assert_called_once_with(
                model="test-model",
                base_url="https://test.api",
                api_key="sk-test",
                temperature=0.5,
                max_tokens=512,
            )

    def test_llm_service_init_explicit(self):
        """Explicit model param overrides config-based creation."""
        mock_model = MagicMock()
        svc = LLMService(model=mock_model)
        assert svc._model is mock_model


class TestLLMServiceGenerate:
    """Tests for LLMService.generate()."""

    @pytest.mark.asyncio
    async def test_generate_returns_llm_response(self):
        """generate() returns LLMResponse with content, model, tokens."""
        mock_model = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "Hello from LLM"
        mock_result.response_metadata = {"model_name": "test-model"}
        mock_result.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        mock_model.ainvoke = AsyncMock(return_value=mock_result)

        svc = LLMService(model=mock_model)
        resp = await svc.generate([{"role": "user", "content": "Hi"}])

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello from LLM"
        assert resp.model == "test-model"
        assert resp.tokens_in == 10
        assert resp.tokens_out == 5

    @pytest.mark.asyncio
    async def test_generate_extracts_token_usage(self):
        """Token usage extracted from response.usage_metadata."""
        mock_model = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "reply"
        mock_result.response_metadata = {"model_name": "m"}
        mock_result.usage_metadata = {"input_tokens": 42, "output_tokens": 17}
        mock_model.ainvoke = AsyncMock(return_value=mock_result)

        svc = LLMService(model=mock_model)
        resp = await svc.generate([{"role": "user", "content": "x"}])

        assert resp.tokens_in == 42
        assert resp.tokens_out == 17

    @pytest.mark.asyncio
    async def test_generate_propagates_exception(self):
        """Errors from the model are propagated, not swallowed."""
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))

        svc = LLMService(model=mock_model)

        with pytest.raises(RuntimeError, match="API down"):
            await svc.generate([{"role": "user", "content": "x"}])


class TestMessageConversion:
    """Tests for message dict to langchain type conversion."""

    def test_messages_converted_to_langchain_types(self):
        """Dicts become SystemMessage/HumanMessage/AIMessage."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        svc = LLMService(model=MagicMock())
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = svc._convert_messages(messages)

        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)
        assert result[0].content == "Be helpful"
        assert result[1].content == "Hi"
        assert result[2].content == "Hello"


class TestLLMServiceSingleton:
    """Tests for DI get/set pattern."""

    def test_get_llm_service_singleton(self):
        """get_llm_service() returns same instance on repeated calls."""
        mock = MagicMock()
        set_llm_service(mock)
        assert get_llm_service() is mock
        assert get_llm_service() is mock

    def test_set_llm_service(self):
        """set_llm_service(mock) then get returns mock."""
        mock = MagicMock()
        set_llm_service(mock)
        assert get_llm_service() is mock
