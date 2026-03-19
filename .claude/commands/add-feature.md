Scaffold all layers of a new feature at once, then verify with tests and CI.

## Arguments

$ARGUMENTS = feature description

## Workflow

### Step 1: Determine required layers

Ask the user via AskUserQuestion which layers this feature needs. Present as a checklist:

- [ ] **Database migration** — new tables, columns, or indexes?
- [ ] **Service** — new business logic service?
- [ ] **Handler** — new Telegram command, message handler, or callback?
- [ ] **Tests** — always yes unless user explicitly opts out

Also ask:
- Brief description of what each selected layer should do
- Any dependencies on existing services or tables

Wait for answers before proceeding.

### Step 2: Scaffold in dependency order

Invoke the following skills **in order** based on the user's selections. Each skill receives the feature description plus context from previous steps.

**2a. Database migration** (if selected):
Invoke the `/add-migration` skill with the table/column requirements.

**2b. Service** (if selected):
Invoke the `/add-service` skill with the service name and purpose. If a migration was created in 2a, mention the new tables/columns so the service can reference them.

**2c. Handler** (if selected):
Invoke the `/add-handler` skill with the handler type and purpose. If a service was created in 2b, mention it so the handler can wire it in.

### Step 3: Write tests

Invoke the **tester** agent to write tests for all scaffolded code:
- Service tests: happy path, error path, graceful degradation
- Handler tests: mock services, verify `message.answer()` calls, verify error fallback
- Integration between layers if applicable

The tester agent should read the files created in Step 2 to understand what needs testing.

### Step 4: Run CI gate

Invoke the `/check` skill to run the full CI gate (format, lint, typecheck, tests). If anything fails, fix and re-run until all four pass.

### Step 5: Summary

Report to the user:
- File tree of everything created, grouped by layer:
  ```
  Migrations:
    sql/migrations/NNNN_<name>.sql
  Services:
    src/bot/services/<name>.py
  Handlers:
    src/bot/handlers.py (modified)
  Tests:
    tests/test_<name>.py
  ```
- Brief description of what each file does
- Wiring instructions: any manual steps needed (e.g., register handler in app.py, add env vars, run migration)
