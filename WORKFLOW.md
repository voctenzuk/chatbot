# Development Workflow

gstack lifecycle + project-specific domain agents and skills.

## New Feature (full cycle)

```
/office-hours → /plan-ceo-review → /plan-eng-review → /implement → /check → /review → /ship → /document-release
```

### 1. Think — `/office-hours`

Brainstorm before writing code. Two modes:
- **Startup mode** — six forcing questions (demand, status quo, wedge, etc.)
- **Builder mode** — design thinking for side projects / hackathons

Output: design doc saved automatically, fed into planning phase.

### 2. Plan (strategy) — `/plan-ceo-review`

Challenge scope and ambition. Four modes:
- **SCOPE EXPANSION** — dream big
- **SELECTIVE EXPANSION** — hold scope + cherry-pick expansions
- **HOLD SCOPE** — maximum rigor within current boundaries
- **SCOPE REDUCTION** — strip to essentials

### 3. Plan (architecture) — `/plan-eng-review`

Lock the execution plan:
- ASCII diagrams for data flow and state machines
- Edge cases and failure modes
- Test matrix design
- Security concerns
- Performance considerations

### 4. Build — `/implement` or `/add-feature`

`/implement` orchestrates the full feature lifecycle:
1. Clarify requirements (asks questions)
2. Invoke `planner` agent for phased plan
3. Get user approval
4. Execute phases with specialist agents
5. Run tests after each phase
6. Full CI gate via `/check`

`/add-feature` scaffolds all layers at once (migration → service → handler → tests).

**Domain agents** used during build:

| Agent | When |
|---|---|
| `memory-specialist` | Cognee pipeline, episodes, dual-layer memory |
| `llm-pipeline` | LLM service, context building, prompt engineering |
| `telegram-handler` | aiogram 3 handlers, commands, callbacks, middleware |
| `prompt-engineer` | System prompt quality, injection defense, token efficiency |
| `refactor-mover` | Moving modules between packages with backward-compatible shims |
| `tester` | Writing and fixing pytest tests |

**Skills** auto-triggered during build:

| Skill | Triggered when |
|---|---|
| `add-migration` | Feature needs new/modified DB tables — runs before service code |
| `add-service` | New business logic module needed — scaffolds with singleton + tests |
| `add-handler` | New bot command or interaction — scaffolds with conventions |
| `gen-test` | Module created/modified without tests — runs automatically |

### 5. CI gate — `/check`

Runs sequentially, fixes failures, re-runs until green:
```
ruff format --check . → ruff check . → pyright → pytest
```

### 6. Review — `/review`

gstack pre-landing review: SQL safety, LLM trust boundaries, conditional side effects.

Optional additions:
- `/codex` — independent second opinion via OpenAI (review, challenge, or consult mode)
- `security-reviewer` agent — user isolation, input sanitization, secrets exposure
- `prompt-review` skill — auto-triggered after changes to system_prompt.py or context_builder.py

### 7. Ship — `/ship`

Full shipping pipeline:
1. Detect + merge base branch
2. Run tests
3. Review diff
4. Bump VERSION
5. Update CHANGELOG
6. Commit, push, create PR

### 8. Docs — `/document-release`

Post-ship: syncs README, ARCHITECTURE, CONTRIBUTING, CLAUDE.md, CHANGELOG to match what shipped.

---

## Bugfix

```
/investigate → fix → /check → /ship
```

- `/investigate` — systematic root-cause debugging (general)
- `diagnose` skill — bot-specific diagnostics (config, wiring, handlers, memory, LLM)

Use `/investigate` for general errors. Use `diagnose` when the issue is clearly bot-domain
(memory not saving, episodes not switching, LLM not responding, handler not firing).

---

## Refactoring

```
/refactor-phase → /check → /review → /ship
```

Each phase is a separate branch + PR. The `refactor-mover` agent handles file moves
with backward-compatible shims. `verify-imports` skill runs automatically after moves
to catch broken imports before marking done.

---

## Prompt Work

```
Edit system_prompt.py / context_builder.py → prompt-review skill (auto) → /check → /ship
```

`prompt-review` auto-triggers after changes to prompt-related files. Rates 1-10 on:
- Persona quality (Russian naturalness, consistency)
- Context assembly efficiency (token budget)
- Prompt injection defense
- Summary/memory extraction quality

---

## Weekly

```
/retro
```

Analyzes commit history, work patterns, code quality metrics. Team-aware breakdowns.
Persistent history for trend tracking across weeks.

---

## Safety Tools

| Command | What it does | When to use |
|---|---|---|
| `/careful` | Warns before destructive commands (rm -rf, DROP TABLE, force-push) | Touching prod, shared infra |
| `/freeze <dir>` | Restricts edits to one directory | Debugging — prevent accidental changes elsewhere |
| `/guard` | `/careful` + `/freeze` combined | Maximum safety |
| `/unfreeze` | Remove edit restrictions | Done with scoped work |

---

## Branch Conventions

- **`main`** — production
- **`develop`** — integration
- **`feature/<name>`** — new features (from `develop`)
- **`fix/<name>`** — bugfixes (from `develop`, or `main` for hotfixes)
- Squash or rebase merge preferred

## CI Requirements

Every PR must pass: `ruff format --check .` → `ruff check .` → `pyright` → `pytest`

## Commit Messages

Present tense, prefixed: `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`.
Reference issues when applicable: `fix: handle timeout in redis client (#42)`.
