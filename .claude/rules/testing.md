---
paths:
  - "tests/**/*.py"
---

# Testing conventions

Project uses `asyncio_mode = "auto"` (pyproject.toml), but we still decorate with `@pytest.mark.asyncio` for explicitness.

## Structure
- Tests grouped in classes (`TestFeatureName`) per logical area
- One test file per module: `tests/test_<module>.py`
- No shared `conftest.py` -- fixtures live in the test file that needs them
- Use `from __future__ import annotations` in every test file

## Async mock patterns

<important>
Use `AsyncMock` for any method that is `await`-ed. Use `MagicMock` for sync attributes and objects.
Never use `MagicMock` where `AsyncMock` is needed -- the test will return a coroutine-never-awaited warning and silently pass with wrong assertions.
</important>

```python
# Async service methods
manager = AsyncMock()
manager.process_user_message = AsyncMock(return_value=mock_result)

# Sync data objects / specs
mock_episode = MagicMock()
mock_episode.id = "ep_1"
```

## Fixture patterns
- `autouse=True` fixtures for singleton teardown (reset global state between tests)
- `yield` inside `with patch(...)` for fixtures that must mock + cleanup
- For complex dependencies (e.g. `DatabaseClient`), create a `MagicMock(spec=RealClass)` then assign async closures to its methods instead of relying on `AsyncMock` auto-spec

<important>
Always patch at the import location, not the definition: `patch("bot.handlers.get_episode_manager_service")` not `patch("bot.services.episode_manager.get_episode_manager_service")`.
</important>

## Custom mock objects
Prefer purpose-built mock classes (`MockMessage`, `MockEmbeddingProvider`) over deeply nested `MagicMock` when the real object has complex structure (e.g., aiogram `Message`). Keep them minimal -- only the fields tests actually touch.

## Error / edge-case tests
- Test both the happy path and error propagation (`pytest.raises`)
- For handler-level tests, verify graceful degradation: the bot should still call `message.answer()` even when services fail
