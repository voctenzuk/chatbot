---
paths:
  - "**/*.py"
---

# Python style & conventions

## Toolchain

CI gate: `uvx ruff format --check .` → `uvx ruff check .` → `uvx pyright` → `uv run pytest` (in that order).
Always run `uvx ruff check --fix . && uvx ruff format .` before committing.

## Ruff

Enabled rules: F, E/W, I, UP, B, SIM, ASYNC, C4, PIE, T20, RUF. Config in pyproject.toml.

<important>
- Line length: **100** characters
- `target-version = "py313"` — use modern syntax: `X | Y` unions, `list[str]` not `List[str]`
- `T20` catches stray `print()` in production code. Allowed in `tests/` and `scripts/`
- Import sorting via `I` — `known-first-party = ["bot"]`
- `raise ... from err` inside `except` blocks (B904)
- Do NOT enable: ANN (pyright does it better), D (docstrings — for libraries), N (naming — conflicts with frameworks)
</important>

## Pyright

Mode: `standard` (explicit in pyproject.toml). `useLibraryCodeForTypes = false`.

<important>
- All public functions MUST have type annotations (params + return)
- Use `str | None` not `Optional[str]`. Use `X | Y` union syntax
- `async def` return type is the unwrapped value (`-> str`), never `-> Coroutine[...]`
- Never blanket `# type: ignore` — use `# pyright: ignore[reportSpecificRule]` with a why-comment
- Do NOT use `cast()` to hide real mismatches — fix the types
- Do NOT use `from __future__ import annotations` — not needed on 3.13+, breaks pydantic runtime type evaluation
- `reportMissingTypeStubs = false` — many deps lack stubs, this is expected
</important>

### Type narrowing
- `isinstance()` for unions, `x is not None` for Optional
- `asyncio.gather(return_exceptions=True)` returns `list[T | BaseException]` — narrow with `isinstance` before use

## Async patterns

Async-first codebase (aiogram + LangChain).
- `async def` for all I/O. LangChain: always `.ainvoke()` / `.astream()`, never sync `.invoke()`
- Wrap untyped third-party calls in typed service boundaries

## Imports

stdlib → third-party → local (`bot.*`). Handled by ruff `I` rule.
Use absolute imports (`from bot.conversation.episode_manager import EpisodeManager`).

## Project patterns

- Pydantic v2 settings: `BaseSettings` with `SettingsConfigDict(env_file=".env", extra="ignore")`
- Value objects: `@dataclass(frozen=True)` or `@dataclass`
- Logging: `loguru.logger` with `{}` placeholders: `logger.info("msg {}", var)`
- Commits: present tense, prefixed (`feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`)
