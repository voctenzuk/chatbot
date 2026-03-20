---
paths:
  - "src/bot/llm/**"
  - "src/bot/conversation/summarizer.py"
  - "src/bot/conversation/context_builder.py"
---

# LangChain integration

Uses `langchain-core>=0.2`, `langchain-openai>=0.1`, `langgraph>=0.2`.

## Imports

<important>
Always import from split packages, never legacy monolithic `langchain`:
- `from langchain_core.messages import SystemMessage, HumanMessage, AIMessage`
- `from langchain_openai import ChatOpenAI`
Do NOT use `from langchain.chat_models import ...` or `from langchain.schema import ...`
</important>

## Async invocation

<important>
Always `await model.ainvoke(messages)` — never sync `model.invoke()`.
The bot runs on asyncio; blocking calls freeze the event loop.
</important>

## LLMService pattern
- `LLMService` (`src/bot/llm/service.py`) — single entry point for LLM calls
- Accepts `BaseChatModel` for DI/testing, defaults to `ChatOpenAI` from settings
- Returns `LLMResponse(content, model, tokens_in, tokens_out)`
- Token usage from `result.usage_metadata` and `result.response_metadata`

## Summarizer
- `Summarizer` (`src/bot/conversation/summarizer.py`) uses a separate `LLMProvider` protocol (raw prompt → string)
- Does NOT go through `LLMService`. Do not mix the two patterns

## ChatOpenAI configuration
- `base_url` may point to non-OpenAI providers. Do not rely on OpenAI-specific features without checking compatibility

## Testing
Inject mock `BaseChatModel` via `LLMService(model=mock)`. Never call real APIs in tests.
