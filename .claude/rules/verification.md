---
paths:
  - "**/*.py"
  - "tests/**"
---

# Verification Before Completion

Never claim a task is done without fresh verification evidence.

## Rules

1. **Run the command. Read the output. THEN claim the result.**
   - "Tests pass" requires actual `uv run pytest` output in this session
   - "Types check" requires actual `uvx pyright` output in this session
   - "Linting clean" requires actual `uvx ruff check .` output in this session

2. **Stale evidence is not evidence.** If you ran tests 10+ tool calls ago, run them again before claiming success.

3. **No weasel words.** Never say "should work", "looks correct", "I believe this passes". Either you ran it and it passed, or you haven't verified yet — say so.

4. **After fixing a failing test:** re-run the ENTIRE test suite, not just the fixed test. Fixes frequently break other tests.

5. **After any Edit/Write to production code:** run at minimum `uvx ruff check --fix <file> && uvx ruff format <file>` (the PostToolUse hook does this automatically, but verify if unsure).
