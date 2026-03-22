---
paths:
  - "tests/**/*.py"
  - "tests/conftest.py"
---

# Testing conventions

## Structure
- Flat `tests/test_<module>.py`, one per module. Group with `class TestFeatureName`
- Shared fixtures in `tests/conftest.py` (mock_db, mock_llm, mock_delivery). File-local only for module-specific needs
- `asyncio_mode = "auto"` — `@pytest.mark.asyncio` optional but allowed for clarity

## Fixtures
- **Factory functions** (`_make_pipeline(**overrides)`) for objects with variations — preferred pattern
- **`@pytest.fixture(autouse=True)`** for singleton resets. Do NOT use `setup_method/teardown_method`
- `yield` for teardown, never `addfinalizer`. Scope defaults to `function`
- Max fixture chain depth: 2 levels

## Mocking

<important>
- `AsyncMock` for any `await`-ed method. Never `MagicMock` for async — returns coroutine-never-awaited warning and silently passes
- `spec=RealClass` on mocks to catch attribute typos
- Patch at the **import location**, not the definition: `patch("bot.handlers.get_service")` not `patch("bot.conversation.episode_manager.get_service")`
</important>

- Mock at boundaries (LLM API, Supabase, Redis, Cognee), NOT internal methods between own classes
- Assert outcomes, NOT call order. `assert_has_calls(..., any_order=False)` is an anti-pattern
- If `mock.a.b.c.return_value` — test is over-coupled, refactor
- Purpose-built mock classes (`MockMessage`) for complex types (aiogram `Message`), not deeply nested MagicMock

## Parametrize
- Use when 3+ tests have identical structure with different data. `pytest.param(..., id="name")` for readable output
- Do NOT parametrize if cases need different setup/assertion logic

## Markers
- `@pytest.mark.slow` — tests >2s, skip with `pytest -m "not slow"`
- `@pytest.mark.integration` — tests hitting real external services

## Coverage
- CI gate: 80% line + branch coverage (`pytest --cov=bot --cov-branch`)
- Exclude: `__main__.py`, `if TYPE_CHECKING:` blocks. Do NOT chase 100%

## Anti-patterns — do NOT
- Test private methods (`_internal()`) — test through public API
- Use `asyncio.sleep()` for sync — `await` tasks explicitly or use `asyncio.Event`
- Write tests longer than 30 lines without extracting helpers
- Share mutable state between tests
- Assert broadly (`assert "error" in str(result)`) — assert specific types/messages
