Orchestrate one phase of the architectural refactoring. Each phase is a standalone, shippable unit of work.

## Arguments

$ARGUMENTS = phase number and description (e.g., "1 — Extract ChatPipeline and Ports")

## Refactoring Phases Overview

| Phase | Description | Key agents |
|---|---|---|
| 1 | Extract ChatPipeline + Ports (3 protocols) | telegram-handler, tester |
| 2 | Fix leaky abstractions (db._client access) | tester |
| 3a | Domain grouping: conversation/ | refactor-mover, memory-specialist |
| 3b | Domain grouping: memory/ | refactor-mover, memory-specialist |
| 3c | Domain grouping: llm/ | refactor-mover, llm-pipeline |
| 3d | Domain grouping: media/ + infra/ | refactor-mover |
| 3e | Kill services/__init__.py | refactor-mover |
| 4 | Web adapter (future) | telegram-handler |

## Workflow

### Step 1: Create branch

```bash
git checkout -b refactor/phase-{N}-{description}
```

### Step 2: Plan the phase

Invoke the **planner** agent with the phase description. The planner will:
- Read relevant source files
- Identify all files to modify/create
- Produce acceptance criteria
- Define the test gate

Present the plan to the user for approval.

### Step 3: Execute

Based on the phase, delegate to the appropriate agents:

**Phase 1 (ChatPipeline + Ports):**
1. Create `src/bot/ports.py` — 3 Protocol classes (LLMPort, MemoryPort, MessageDeliveryPort)
2. Create `src/bot/chat_pipeline.py` — extract logic from handlers.py::chat()
3. Refactor `src/bot/handlers.py` — thin wrapper calling ChatPipeline
4. Refactor `src/bot/services/proactive_scheduler.py` — use MessageDeliveryPort
5. Use **telegram-handler** agent for handler refactoring
6. Use **tester** agent for test updates

**Phase 2 (Fix leaky abstractions):**
1. Add missing methods to `src/bot/services/db_client.py`
2. Replace `db._client.table()` calls in artifact_service.py (6 places)
3. Replace `db._client.table()` calls in proactive_scheduler.py (1 place)
4. Direct implementation + **tester** agent

**Phase 3a-3e (Domain grouping):**
1. Use **refactor-mover** agent for each module relocation
2. Use domain specialists to validate (memory-specialist, llm-pipeline)
3. Create `__init__.py` with public API for each new package
4. Invoke `/verify-imports` skill after each sub-phase

### Step 4: Validate

Run the full CI gate:
```bash
uvx ruff format --check .
uvx ruff check .
uv run pyright
uv run pytest
```

Fix any failures. Repeat until all pass.

### Step 5: Review

Launch both reviewers in parallel:
- **reviewer** agent — correctness, types, async, conventions
- **security-reviewer** agent — data isolation, injection, auth

### Step 6: Ship

If review is clean:
1. Stage and commit: `refactor: phase {N} — {description}`
2. Ask user: push and create PR?

## Rules

- Each phase must be independently shippable (green CI, no broken imports)
- Backward-compatible shims are OK during migration (cleaned up in phase 3e)
- Do NOT combine phases — one PR per phase
- If a phase is too large, split it and create sub-phases
- Always run `/verify-imports` after moving files
