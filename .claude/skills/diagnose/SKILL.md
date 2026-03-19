---
name: diagnose
description: When the bot misbehaves, crashes, or produces unexpected responses
context: fork
disable-model-invocation: true
---

# Diagnose

Systematically diagnose bot issues by checking configuration, service health, and common failure modes.

## Arguments

$ARGUMENTS = symptom description (e.g., "bot not responding", "memory not working", "LLM errors", "episodes not switching"). If empty, run full diagnostic.

## Steps

1. **Config check** — Read `src/bot/config.py` and verify:
   - All required env vars are documented and have sensible defaults
   - Optional services (Redis, Cognee, Supabase) degrade gracefully when missing
   - LLM settings (model, base_url, temperature, max_tokens) are reasonable

2. **Service wiring check** — Read `src/bot/app.py` and verify:
   - All singletons are initialized before `start_polling()`
   - Initialization order is correct (config → DB → services → handlers)
   - Shutdown/cleanup hooks exist where needed

3. **Handler flow check** — Read `src/bot/handlers.py` and verify:
   - Error handling catches all service-level exceptions
   - Fallback text is always sent to user on error
   - Catch-all handler is registered last
   - No blocking (sync) calls in async handlers

4. **Memory pipeline check** (if symptom relates to memory):
   - Read `cognee_memory_service.py` — verify add/cognify/search flow
   - Check dataset scoping (`tg_user_{user_id}`)
   - Verify `_pending_datasets` is cleared after cognify
   - Check cognify scheduling (not called too often or never)

5. **Episode lifecycle check** (if symptom relates to episodes):
   - Read `episode_manager.py` and `episode_switcher.py`
   - Verify time-gap and topic-shift thresholds
   - Check DB persistence of episodes/messages

6. **LLM integration check** (if symptom relates to responses):
   - Read `llm_service.py` — verify async invocation, error handling
   - Read `context_builder.py` — verify message assembly and token budgeting
   - Check for provider-specific issues (base_url, model compatibility)

7. **Output a diagnostic report**:

```
## Diagnostic Report

### Symptom
[User-reported issue]

### Checks Performed
- [x] Config: OK / ISSUE FOUND
- [x] Service wiring: OK / ISSUE FOUND
- [x] Handler flow: OK / ISSUE FOUND
- [ ] Memory pipeline: SKIPPED (not relevant)
...

### Findings
1. [CRITICAL/WARNING/INFO] description — file:line
   Cause: ...
   Fix: ...

### Recommended Actions
1. ...
```

## Gotchas

- Trying to read `.env` file — denied by permissions; diagnose from code and config defaults only.
- Suggesting fixes without reading the actual code first — always read the relevant source files before proposing changes.
- Assuming services are required when they're optional — Redis, Cognee, and Supabase all degrade gracefully; missing services are not bugs.
- Not checking singleton initialization order in `app.py` — services depend on each other and must be initialized in the correct sequence.
- Conflating "service unavailable" with "service misconfigured" — check whether the service is simply absent (expected) vs. present but broken (actual bug).
