Pre-merge shipping pipeline. Validate, review, and prepare current changes for merge.

## Workflow

### Step 1: Run CI gate

Invoke the `/check` skill to run all four CI checks sequentially:
1. `uvx ruff format --check .` (formatting)
2. `uvx ruff check .` (linting)
3. `uv run pyright` (type checking)
4. `uv run pytest` (tests)

### Step 2: Fix CI failures

If any check failed, fix the issues and re-run all checks from the beginning. Repeat until all four pass. Do not proceed to review until CI is green.

### Step 3: Parallel code review

Launch both review agents **in parallel**:
- Invoke the **reviewer** agent — code review for correctness, types, async, conventions
- Invoke the **security-reviewer** agent — security audit for data leaks, injection, auth, isolation

Wait for both to complete.

### Step 4: Evaluate findings

Collect findings from both reviewers. Categorize them:
- **Blockers** (reviewer: `BLOCK`, security-reviewer: `CRITICAL` or `HIGH`) — must fix before merge
- **Warnings** (reviewer: `WARN`, security-reviewer: `MEDIUM`) — should fix
- **Nits** (reviewer: `NIT`, security-reviewer: `LOW`) — optional

### Step 5: Handle blockers

If blockers were found:
1. Present them to the user via AskUserQuestion
2. Ask: "Should I fix these blockers, or do you want to handle them manually?"
3. If user says fix: address each blocker, then re-run `/check` and re-invoke both reviewers
4. Repeat until no blockers remain

If only warnings/nits were found, present them and ask if the user wants any addressed before proceeding.

### Step 6: Create commit

Once all checks pass and no blockers remain:
1. Run `git status` and `git diff --cached` and `git diff` to see what will be committed
2. Stage relevant files (prefer explicit file names over `git add -A`)
3. Create a commit with a descriptive present-tense message following the project convention (e.g., `feat: add proactive messaging scheduler`)

### Step 7: Push and PR

Ask the user via AskUserQuestion:
- "All checks pass and review is clean. Want me to push and create a PR?"
- If yes: push the branch and create a PR using `gh pr create` with a summary of changes
- If no: stop here and report what was done
