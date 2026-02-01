# Development Workflow

## Overview

This repository follows a **two-agent development model**:
- **Kimi** → Writes code, implements features, fixes bugs
- **Molter** → Reviews code, provides feedback, ensures quality

## Branch Conventions

### Main Branches
- **`main`** — Production-ready code. Only merged via reviewed PRs.
- **`develop`** — Integration branch for features. Merges into `main` when stable.

### Feature Branches
- Format: `feature/<short-description>`
- Example: `feature/memory-context-window`
- Created from: `develop`
- Merged to: `develop`

### Bugfix Branches
- Format: `fix/<short-description>`
- Example: `fix/redis-connection-timeout`
- Created from: `develop` (or `main` for hotfixes)
- Merged to: `develop` (or `main` for hotfixes)

## Pull Request Workflow

### 1. Kimi Creates PR
- Branch follows naming convention above
- PR uses the template in `.github/pull_request_template.md`
- All CI checks must pass before requesting review

### 2. Molter Reviews
- Review for code quality, logic correctness, test coverage
- Approval required before merge
- Changes requested → Kimi addresses, pushes updates

### 3. Merge Requirements
- [ ] All CI checks passing (ruff, pyright, pytest)
- [ ] Molter approval
- [ ] No merge conflicts
- [ ] Squash or rebase merge preferred for clean history

## CI Requirements

Every PR must pass:
- `ruff format --check` — Code formatting
- `ruff check` — Linting
- `pyright` — Type checking
- `pytest` — Unit tests

## Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Be concise but descriptive
- Reference issues when applicable: "Fix #123: Handle edge case in ..."

## Code Style

- Python 3.12+ features encouraged
- Type hints required on public functions
- Docstrings for modules and complex functions
- 100 character line length (per `pyproject.toml`)

## Release Process

1. Update version in `pyproject.toml`
2. Create PR from `develop` → `main`
3. Full review by Molter
4. Tag release after merge: `git tag v0.x.x`
5. Push tags: `git push origin --tags`
