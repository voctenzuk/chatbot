Orchestrate the full lifecycle of implementing a feature. You are a lightweight coordinator — delegate all code work to specialist agents, never write code yourself.

## Arguments

$ARGUMENTS = feature description

## Workflow

### Step 1: Clarify & route

Ask the user 2-3 clarifying questions via AskUserQuestion. Max 3 questions — make informed guesses for the rest.

- What exactly should this feature do? (confirm scope)
- Which layers are involved? (handler, service, DB, memory, LLM, config)
- Any constraints or edge cases?

Wait for answers before proceeding.

### Step 2: Plan

Invoke the **planner** agent with the feature description + user's clarifications. The planner will analyze the codebase and produce a phased plan with: files to create/modify, acceptance criteria, and test requirements per phase.

### Step 3: Approve

Present the plan to the user via AskUserQuestion:
- The phased plan with files and acceptance criteria
- Estimated agent routing per phase (from the routing table below)
- Ask: "Ready to proceed? Any changes?"

Do NOT proceed until the user approves.

### Step 4: Execute phases

For each phase in the approved plan, run the **implement → verify → review** loop:

#### 4a. Dispatch specialist agent

Pick the agent by topic. Paste the FULL phase description + acceptance criteria + relevant file paths directly into the agent prompt — never reference a plan file.

**Routing table:**

| Phase topic | Agent | When to use |
|---|---|---|
| Handlers, commands, callbacks, middleware, filters | **telegram-handler** | Touching `handlers.py`, `app.py`, `adapters/` |
| LLM service, LangChain chains, model config | **llm-pipeline** | Touching `llm/`, `LLMService`, tool definitions |
| Memory pipeline, episodes, context builder | **memory-specialist** | Touching `memory/`, `conversation/`, episodes |
| System prompt, persona, few-shot examples | **prompt-engineer** | Touching `system_prompt.py`, `character.py` |
| DB migrations, Supabase client, infra | Do it yourself | Touching `infra/`, `config.py`, migrations |
| Module relocation, import updates | **refactor-mover** | Moving files between packages |
| General services, config, wiring, multiple layers | Do it yourself | Cross-cutting changes, `wiring.py` |

If a phase touches multiple domains, split it or handle the cross-cutting part yourself and delegate domain-specific parts to agents.

#### 4b. Handle agent status

The agent will report one of:

| Status | Your action |
|---|---|
| **Done** | Proceed to verification (4c) |
| **Done with concerns** | Read concerns. If correctness/scope issue → address before verify. If observation → note and proceed. |
| **Needs context** | Provide missing context, re-dispatch same agent |
| **Blocked** | Assess: context problem → re-dispatch with context. Task too large → split. Plan wrong → ask user. |

If the agent is **blocked after 2 re-dispatches**, STOP and ask the user how to proceed.

#### 4c. Verify

Run `uv run pytest` and read the output.

- **Tests pass** → proceed to review (4d)
- **Tests fail** → invoke **tester** agent with the failure output to fix. Re-run tests.
- **Tests fail 3 times** → STOP. Report what's failing and ask the user. Do NOT attempt a 4th fix without user input or a fundamentally different approach.

#### 4d. Review

Invoke the **reviewer** agent with this prompt:

> Review the changes from the current phase. Run `git diff HEAD~N` (where N = number of commits in this phase) to see only this phase's changes. Focus on: correctness, async discipline, type safety, user isolation, error handling. Report BLOCK/WARN/NIT findings.

Handle review results:

- **APPROVE** → commit phase, move to next phase
- **REQUEST_CHANGES with BLOCKers** → dispatch the original specialist agent with the BLOCK findings pasted in full. After fix, re-run verify (4c) then re-review (4d). **Max 2 review iterations per phase** — if still blocked, ask user.
- **WARN/NIT only** → fix WARNs yourself if trivial (1-2 line changes), skip NITs. Proceed.

#### 4e. Commit phase

Stage and commit the phase: `git add <specific files> && git commit -m "feat: <phase description>"`

Briefly report to the user: what was done, files changed, any decisions made.

### Step 5: Full CI gate

After all phases, invoke the `/check` skill. If anything fails, fix and re-run until all four checks pass (format, lint, typecheck, tests).

### Step 6: Summary

Report to the user:
- What was implemented (files created/modified, grouped by phase)
- Tests added and coverage
- Decisions made during implementation
- Any WARN/NIT items deferred from review
- Suggested follow-ups or known limitations

## Rules

- **Never write code yourself** except for trivial cross-cutting glue (config, wiring, 1-2 line fixes).
- **Paste full context into agent prompts** — acceptance criteria, file paths, relevant code snippets. Agents start with zero context.
- **Fresh verification evidence only** — "tests pass" means you ran `uv run pytest` and read the output in this session. See `.claude/rules/verification.md`.
- **3-strike rule** — 3 failed fixes on the same issue → STOP and ask user. See `.claude/rules/debugging.md`.
- **Max 2 review iterations per phase** — if reviewer still finds BLOCKers after 2 fix cycles, escalate to user.
- **Commit after each phase** — not at the end. Each phase should be independently revertable.
