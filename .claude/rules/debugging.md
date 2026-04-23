---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Systematic Debugging

No fixes without root-cause investigation. Adapted from gstack /investigate and Superpowers methodology.

## Four Phases

### Phase 1: Investigate
- Reproduce the failure (run the failing test/command)
- Read the full stack trace and error message
- Trace data flow from input to failure point
- Check recent changes (`git diff`, `git log --oneline -10`)

### Phase 2: Analyze
- Identify the layer: handler → pipeline → service → DB → external API
- Check: is this async-related? (missing await, wrong event loop, race condition)
- Check: is this data-related? (wrong user_id scoping, stale cache, missing migration)
- Check: is this config-related? (missing env var, wrong model name, disabled service)

### Phase 3: Hypothesize
- Form ONE specific hypothesis before changing code
- State the hypothesis explicitly: "The failure is because X, and fixing Y should resolve it"
- Predict what the fix will change in the output

### Phase 4: Implement & Verify
- Make the minimal change to test the hypothesis
- Run the failing test/command again
- If it passes, run the full test suite to check for regressions
- If it fails, return to Phase 1 with new evidence

## Three-Strike Rule

After 3 failed fix attempts on the same issue:
1. STOP fixing
2. Summarize what was tried and why it failed
3. Ask whether the architecture or approach needs rethinking
4. Do NOT try a 4th fix without user approval or a fundamentally different approach
