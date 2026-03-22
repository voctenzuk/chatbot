---
paths:
  - "**/*.py"
---

# Python style & conventions

## Toolchain

CI gate: `uvx ruff format --check .` → `uvx ruff check .` → `uvx pyright` → `uv run pytest` (in that order).
Always run `uvx ruff check --fix . && uvx ruff format .` before committing.
Full linter/type-checker config lives in `pyproject.toml` — that is the source of truth, not this file.

## Code style

<important>
- Python 3.13+, line length 100
- Modern syntax: `str | None` not `Optional[str]`, `list[str]` not `List[str]`
- No `from __future__ import annotations` — not needed on 3.13+, breaks pydantic runtime type evaluation
- Always timezone-aware datetimes: `datetime.now(tz=UTC)`, never bare `datetime.now()`
- `raise ... from err` inside `except` blocks
- No `print()` in production code (allowed in `tests/` and `scripts/`)
</important>

## Type annotations

<important>
- All public functions MUST have type annotations (params + return)
- `async def` return type is the unwrapped value (`-> str`), never `-> Coroutine[...]`
- Never blanket `# type: ignore` — use `# pyright: ignore[reportSpecificRule]` with a why-comment
- Do NOT use `cast()` to hide real mismatches — fix the types
- Match statements must be exhaustive — missing cases are errors
- Access `_private` members only within the owning class; expose via public methods/properties
- Narrow types explicitly: `isinstance()` for unions, `x is not None` for optionals
</important>

## Imports

<important>
- Order: stdlib → third-party → local (`bot.*`). Enforced by ruff isort
- Use absolute imports: `from bot.conversation.episode_manager import EpisodeManager`
- All imports MUST be at top-level unless there is a specific reason:
  - **Allowed lazy:** optional deps in `try/except ImportError` (e.g. `mem0`, `openai`, `supabase`); composition root (`wiring.py`)
  - **Never lazy:** stdlib (`re`, `os`, `uuid`), project deps (`numpy`, `langchain_*`), internal `bot.*` without proven circular deps
- Move type-only imports behind `if TYPE_CHECKING:`
- If you suspect circular deps, try top-level import first — only make lazy if it actually fails
- Avoid `global` — prefer mutable container pattern (`dict` holder) for module-level singletons
- Use lowercase for optional-import availability flags: `_mem0_available`, not `MEM0_AVAILABLE`
</important>

## Async patterns

- `async def` for all I/O
- LangChain: always `.ainvoke()` / `.astream()`, never sync `.invoke()`
- Wrap untyped third-party calls in typed service boundaries

## Project patterns

- Pydantic v2 settings: `BaseSettings` with `SettingsConfigDict(env_file=".env", extra="ignore")`
- Value objects: `@dataclass(frozen=True)` or `@dataclass`
- Logging: `loguru.logger` with `{}` placeholders: `logger.info("msg {}", var)`
- Commits: present tense, prefixed (`feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`)
