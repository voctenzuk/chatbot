# Coding agent — minimal workflow

> Coding agents: read this file first and follow it.

## Request template
When you ask for code changes, include:
- Goal
- Acceptance criteria
- Constraints (libs/arch) + any file/structure preferences

## Rules of engagement
- Create a new branch and **push early** so progress is visible.
- Before opening a PR: run `ruff format .`, `ruff check .`, `pytest`.
- Open a PR to `main` with `Closes #...` and leave a short summary comment.
