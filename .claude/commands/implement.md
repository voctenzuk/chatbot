Orchestrate the full lifecycle of implementing a feature. You are a lightweight coordinator — delegate all code work to specialist agents, never write code yourself.

## Arguments

$ARGUMENTS = feature description

## Workflow

### Step 1: Clarify requirements

Ask the user 2-4 clarifying questions using AskUserQuestion:
- What exactly should this feature do? (confirm scope)
- Are there constraints or dependencies on existing services?
- Which layers are involved (handler, service, DB, memory, LLM)?
- Any edge cases or error scenarios to handle?

Wait for answers before proceeding.

### Step 2: Create implementation plan

Invoke the **planner** agent with the feature description and the user's clarifications. The planner will analyze the codebase and produce a phased plan with files, tests, and acceptance criteria for each phase.

### Step 3: Approve the plan

Present the planner's output to the user via AskUserQuestion. Ask:
- Does this plan look right?
- Any phases to add, remove, or reorder?
- Ready to proceed?

Do NOT proceed until the user approves.

### Step 4: Execute phases

For each phase in the approved plan, pick the right specialist agent:

| Phase topic | Agent to invoke |
|---|---|
| Cognee, episodes, memory models, context builder | **memory-specialist** |
| Prompt changes, system prompt, context assembly for LLM | **prompt-engineer** |
| Everything else (handlers, services, config, general code) | Do it yourself using Edit/Write tools |

After each agent completes its phase:
1. Run `uv run pytest` to verify tests pass
2. If tests fail, have the **tester** agent fix them before moving on
3. Briefly report phase completion to the user

### Step 5: Full CI check

After all phases are done, invoke the `/check` skill to run the full CI gate (format, lint, typecheck, tests). If anything fails, fix and re-run until all four pass.

### Step 6: Summary

Report to the user:
- What was implemented (files created/modified)
- Tests added
- Any decisions made during implementation
- Suggested follow-ups or known limitations
