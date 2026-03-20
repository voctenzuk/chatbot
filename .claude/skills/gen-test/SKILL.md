---
name: gen-test
description: Generate a pytest test file for a given module, following project conventions
disable-model-invocation: true
---

# Generate Test

Create a comprehensive test file for an existing module.

## Arguments

$ARGUMENTS = module path (e.g., "src/bot/services/memory_cleanup.py")

## Steps

1. Read the target module to understand its public API (classes, async methods, functions).
2. Read 1-2 existing test files in `tests/` to match the project's test style exactly.
3. Create `tests/test_<module_name>.py` following all conventions from `.claude/rules/testing.md`:
   - `from __future__ import annotations` as the first import
   - `import pytest` and `from unittest.mock import AsyncMock, MagicMock, patch`
   - `@pytest.mark.asyncio` on all async tests
   - `class TestClassName` grouping per logical area
   - `@pytest.fixture(autouse=True)` that resets the module singleton after each test
   - `AsyncMock` for any `await`-ed method, `MagicMock` for sync attributes
   - Patch at the **import location**, not the definition
   - Purpose-built mock classes for complex objects (e.g., aiogram `Message`)
   - One happy-path + one error-path test per public method
4. Run `uvx ruff check --fix && uvx ruff format` on the test file.
5. Run `uv run pytest tests/test_<module_name>.py -v` to verify tests pass.
6. Print summary of tests created and coverage of public API.
