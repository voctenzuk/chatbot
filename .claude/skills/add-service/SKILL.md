---
name: add-service
description: >
  Scaffold a new async service module with singleton wiring and test stubs.
  Use when asked to "add service", "new service", or "create module for X".
  Proactively use when a feature requires new business logic that doesn't fit into
  an existing service — scaffold the service before implementing the logic.
disable-model-invocation: true
---

# Add Service

Scaffold a new service module following project conventions.

## Arguments

$ARGUMENTS = service name and purpose (e.g., "notification_service — sends push notifications")

## Steps

1. Parse the service name from arguments. Convert to snake_case if needed.
2. Read 1-2 existing services in `src/bot/services/` to match the current code style exactly.
3. Create `src/bot/services/<name>.py` with:
   - `from __future__ import annotations` as the first import
   - Stdlib → third-party → local import ordering
   - Loguru logger: `from loguru import logger`
   - Async class with type-annotated public methods (args + return)
   - `str | None` union syntax, not `Optional`
   - Module-level singleton pattern:
     ```python
     _service: ServiceName | None = None

     def get_service_name() -> ServiceName:
         assert _service is not None, "ServiceName not initialized"
         return _service

     def set_service_name(svc: ServiceName | None) -> None:
         global _service
         _service = svc
     ```
   - Docstring on the class explaining its role
4. Create `tests/test_<name>.py` with:
   - `from __future__ import annotations` as the first import
   - `import pytest` and `from unittest.mock import AsyncMock, MagicMock, patch`
   - `@pytest.mark.asyncio` on all async tests
   - `class TestServiceName` grouping
   - `@pytest.fixture(autouse=True)` that resets the singleton after each test
   - One happy-path test stub
   - One error-path test stub using `pytest.raises`
   - Patch at the import location, not the definition
5. Run `uvx ruff check --fix` and `uvx ruff format` on both files.
6. Print a summary of what was created and next steps.

## Gotchas

- Using `Optional[X]` instead of `X | None` — this project uses Python 3.12+ union syntax everywhere.
- Forgetting `from __future__ import annotations` as the first import — required in every module for forward references.
- Using sync methods instead of async — all service methods must be `async def`.
- Forgetting to add singleton teardown in test fixtures — use `@pytest.fixture(autouse=True)` to reset the singleton after each test.
- Patching at the definition location instead of the import location in tests — always patch where the name is looked up, not where it's defined.
