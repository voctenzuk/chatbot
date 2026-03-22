---
name: gen-test
description: >
  Generate a pytest test file for a given module, following project conventions.
  Use when asked to "write tests", "add tests", "cover this module", or "gen test".
  Proactively use after writing or modifying a service/handler that adds new public
  methods without corresponding tests. If you just created or changed a module and
  there is no test file for it — run this skill automatically.
disable-model-invocation: true
---

# Generate Test

Create a comprehensive test file for an existing module.

## Arguments

$ARGUMENTS = module path (e.g., "src/bot/services/memory_cleanup.py")

## Steps

1. Read the target module to understand its public API (classes, async methods, functions).
2. Read `tests/conftest.py` for shared fixtures. Read 1-2 existing test files in `tests/` to match style.
3. Create `tests/test_<module_name>.py` following all conventions from CLAUDE.md Testing Rules:
   - `from __future__ import annotations` as the first import
   - `import pytest` and `from unittest.mock import AsyncMock, MagicMock, patch`
   - `class TestClassName` grouping per logical area
   - **Reuse shared fixtures** from `conftest.py` (mock_db, mock_llm, mock_delivery) before creating local ones
   - `@pytest.fixture(autouse=True)` for singleton resets. Never `setup_method/teardown_method`
   - `AsyncMock` for any `await`-ed method, `MagicMock(spec=RealClass)` for sync attributes
   - Patch at the **import location**, not the definition
   - Factory functions (`_make_pipeline(**overrides)`) for objects with parameter variations
   - Purpose-built mock classes for complex objects (e.g., aiogram `Message`)
   - One happy-path + one error-path test per public method
   - Use `@pytest.mark.parametrize` with `pytest.param(..., id="name")` when 3+ tests share identical structure
   - Mark slow tests (>2s) with `@pytest.mark.slow`, integration tests with `@pytest.mark.integration`
   - Assert outcomes, NOT call order. Never `assert_has_calls(..., any_order=False)`
   - Keep each test under 30 lines — extract helpers for setup
4. Run `uvx ruff check --fix && uvx ruff format` on the test file.
5. Run `uv run pytest tests/test_<module_name>.py -v` to verify tests pass.
6. Print summary of tests created and coverage of public API.
