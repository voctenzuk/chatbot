Run all CI checks sequentially. Fix any failures and re-run until everything passes.

1. Run `uvx ruff format --check .` (formatting)
2. Run `uvx ruff check .` (linting)
3. Run `uv run pyright` (type checking)
4. Run `uv run pytest` (tests)

After all four checks complete:

- If everything passed, report a short summary: which checks ran and that all passed.
- If any check failed, fix the issues, then re-run **all checks from the beginning** to make sure fixes did not introduce new problems. Repeat until all four pass.

Do not skip any step. Do not run checks in parallel.
