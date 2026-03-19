---
paths:
  - "**/*.py"
---

# Python style & conventions

## Toolchain

CI runs `ruff format --check`, `ruff check`, `pyright`, `pytest` (in that order).
Pre-commit: ruff fix + ruff-format. Always run `ruff check --fix && ruff format` before committing.

## Ruff

<important>
Line length is **100** characters (`tool.ruff.line-length = 100`).
No `[tool.ruff.lint]` section exists -- ruff uses **default rules only**: `E4, E7, E9, F` (pycodestyle subset + pyflakes). Do NOT add `# noqa` for rules outside this set.
Ruff-format handles all formatting (double quotes, 4-space indent, trailing commas). Never fight the formatter.
</important>

## Pyright

<important>
Pyright runs in **basic** mode (no `typeCheckingMode` set). Type-annotate all public function signatures (args + return). Use `str | None` union syntax, not `Optional[str]`. Use `from __future__ import annotations` at top of every module for forward-ref support.
</important>

## Async patterns

This is an async-first codebase (aiogram + LangChain).
- Use `async def` for any I/O function; call with `await`.
- LangChain models: always use `.ainvoke()` / `.astream()`, never sync `.invoke()`.
- pytest: `asyncio_mode = "auto"` -- just write `async def test_...`, no decorator needed.

## Imports

stdlib -> third-party -> local (`bot.*`). One blank line between groups.
Ruff-format + isort defaults handle ordering. Use absolute imports (`from bot.services.x import Y`).

## Pydantic v2

Settings use `pydantic-settings`: subclass `BaseSettings` with `SettingsConfigDict(env_file=".env", extra="ignore")`. Fields use `str | None = None` with plain type annotations, not `Field(...)` unless validation is needed.

## Project patterns

- Value objects: `@dataclass(frozen=True)` or `@dataclass`.
- Singletons: `_service: T | None = None` + `get_service() -> T` / `set_service(T | None)` pair for DI/testing.
- Logging: use `loguru.logger` (not stdlib `logging`). Use `{}` placeholders: `logger.info("msg {}", var)`.

## Commits

Present tense, prefixed: `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`.
Reference issues when applicable: `fix: handle timeout in redis client (#42)`.
Branches: `feature/<desc>` or `fix/<desc>` from `develop`.
