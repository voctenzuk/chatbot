---
paths:
  - "src/bot/services/llm_service.py"
  - "src/bot/services/summarizer.py"
  - "src/bot/services/context_builder.py"
---

# LangChain integration

This project uses `langchain-core>=0.2`, `langchain-openai>=0.1`, and `langgraph>=0.2`.

## Imports

<important>
Always import from the split packages, never from the legacy monolithic `langchain`:
- `from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage`
- `from langchain_core.language_models.chat_models import BaseChatModel`
- `from langchain_openai import ChatOpenAI`

Do NOT use `from langchain.chat_models import ...` or `from langchain.schema import ...` -- removed in 0.2+.
</important>

## Message formatting

Messages flow as `list[dict[str, str]]` (`{"role": "system"|"user"|"assistant", "content": ...}`) and are converted in `LLMService._convert_messages()`: `"system"` -> `SystemMessage`, `"assistant"` -> `AIMessage`, else -> `HumanMessage`.

## Async invocation

<important>
Always use `await model.ainvoke(messages)` -- never synchronous `model.invoke()`.
The bot runs on asyncio (aiogram); blocking calls freeze the event loop.
</important>

## LLMService pattern

`LLMService` (`src/bot/services/llm_service.py`) is the single entry point for LLM calls.
- Accepts `BaseChatModel` for DI/testing, defaults to `ChatOpenAI` from settings.
- Returns `LLMResponse(content, model, tokens_in, tokens_out)` dataclass.
- Token usage from `result.usage_metadata` and `result.response_metadata`.
- Global singleton via `get_llm_service()` / `set_llm_service()`.

## Summarizer

`Summarizer` (`src/bot/services/summarizer.py`) uses a separate `LLMProvider` protocol (raw prompt in, string out) -- does NOT go through `LLMService`. Do not mix the two patterns.

## ChatOpenAI configuration

`base_url` may point to non-OpenAI providers (currently `kimi-k2p5`). Do not rely on OpenAI-specific features (function calling, JSON mode) without checking provider compatibility.

```python
ChatOpenAI(model=settings.llm_model, base_url=settings.llm_base_url,
           api_key=settings.llm_api_key, temperature=settings.llm_temperature,
           max_tokens=settings.llm_max_tokens)
```

## Testing
Inject a mock `BaseChatModel` via `LLMService(model=mock)` or `set_llm_service()`. Never call real APIs in tests.
