"""LLM service for generating chat responses via LangChain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from loguru import logger

from bot.config import settings

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the LLM."""

    name: str
    args: dict[str, Any]
    id: str = ""


@dataclass(frozen=True)
class LLMResponse:
    """Immutable result of an LLM generation call."""

    content: str
    model: str
    tokens_in: int
    tokens_out: int
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMService:
    """Thin wrapper around a LangChain chat model.

    Accepts either an explicit ``model`` (for tests / advanced usage) or
    builds a ``ChatOpenAI`` from application settings.
    """

    def __init__(self, model: BaseChatModel | None = None) -> None:
        if model is not None:
            self._model = model
            return

        self._model = ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            max_completion_tokens=settings.llm_max_tokens,
        )
        logger.info("LLMService initialised with model={}", settings.llm_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and return a structured response.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            tools: Optional list of tool schemas (OpenAI function format).
                   When provided, the model may return tool_calls.

        Returns:
            ``LLMResponse`` with content, model name, token counts and tool_calls.

        Raises:
            Any exception from the underlying chat model (not swallowed).
        """
        lc_messages = self._convert_messages(messages)

        if tools:
            # Always rebind so callers can pass different tool sets across calls
            active_model = self._model.bind_tools(tools)  # type: ignore[union-attr]
        else:
            active_model = self._model

        result = await active_model.ainvoke(lc_messages)  # type: ignore[union-attr]

        usage: dict[str, Any] = getattr(result, "usage_metadata", {}) or {}
        meta: dict[str, Any] = getattr(result, "response_metadata", {}) or {}

        # Extract tool calls if present
        parsed_tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(result, "tool_calls", None) or []
        for tc in raw_tool_calls:
            parsed_tool_calls.append(
                ToolCall(
                    name=tc.get("name", ""),
                    args=tc.get("args", {}),
                    id=tc.get("id", ""),
                )
            )

        return LLMResponse(
            content=str(result.content),
            model=meta.get("model_name", "unknown"),
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            tool_calls=parsed_tool_calls,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[BaseMessage]:
        """Convert plain dicts to LangChain message objects."""
        result: list[BaseMessage] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])  # type: ignore[arg-type]
                if tool_calls:
                    result.append(AIMessage(content=content, tool_calls=tool_calls))
                else:
                    result.append(AIMessage(content=content))
            elif role == "tool":
                result.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=msg.get("tool_call_id", ""),
                    )
                )
            else:
                result.append(HumanMessage(content=content))
        return result


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Token pricing per 1K tokens (in cents). Update when models/providers change.
_MODEL_COST_PER_1K: dict[str, tuple[float, float]] = {
    # model_name: (input_cents_per_1k, output_cents_per_1k)
    "kimi-k2p5": (0.1, 0.3),
}
_DEFAULT_COST_PER_1K: tuple[float, float] = (0.1, 0.3)


def estimate_cost_cents(model: str, tokens_in: int, tokens_out: int) -> int:
    """Estimate API cost in cents from token counts.

    Uses per-model rates where known, falls back to default.
    Returns 0 for unknown/zero token counts.

    Args:
        model: Model name string (from LLMResponse.model).
        tokens_in: Input token count.
        tokens_out: Output token count.

    Returns:
        Estimated cost in cents (integer, minimum 0).
    """
    in_rate, out_rate = _MODEL_COST_PER_1K.get(model, _DEFAULT_COST_PER_1K)
    cost = (tokens_in * in_rate + tokens_out * out_rate) / 1000
    return max(int(cost), 0)


# ---------------------------------------------------------------------------
# Dependency-injection helpers (same pattern as other services)
# ---------------------------------------------------------------------------

_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Get or create the global LLMService singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def set_llm_service(service: LLMService | None) -> None:
    """Replace the global LLMService (useful for testing)."""
    global _llm_service
    _llm_service = service
