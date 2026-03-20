Review current changes (staged + unstaged) as a code reviewer.

## Steps

1. Run `git diff` and `git diff --cached` to collect all changes.
2. Read CLAUDE.md and WORKFLOW.md for project conventions.
3. Review every changed hunk for:
   - **Correctness** — logic errors, off-by-one, wrong return types
   - **Type safety** — missing or wrong type hints, pyright-incompatible patterns
   - **Async correctness** — missing `await`, blocking calls inside async functions, sync LangChain calls that should use `ainvoke`/`agenerate`
   - **Error handling** — bare `except`, swallowed exceptions, missing fallback text for users
   - **Test coverage** — new public functions or branches without corresponding tests
4. Check for project-specific pitfalls:
   - aiogram 3 handler patterns (wrong decorator signatures, missing `Router`, using removed aiogram 2.x APIs)
   - Sync calls to LangChain where async equivalents exist (`invoke` vs `ainvoke`)
   - Cognee / memory service calls missing `user_id` isolation (multi-tenant safety)
   - Singleton wiring (`get_*`/`set_*`) used incorrectly or creating circular imports
5. Output a structured review grouped by file. For each finding use a severity tag:
   - **CRITICAL** — will cause a bug or crash at runtime
   - **WARNING** — likely problem, code smell, or missed edge case
   - **SUGGESTION** — style, readability, or minor improvement

## Rules

- Do NOT auto-fix or modify any files. Report findings only.
- If there are no changes, say so and stop.
- Keep feedback actionable: quote the problematic line and explain why it is wrong.
