---
name: prompt-review
description: >
  Review system prompt quality, context assembly efficiency, and injection defense.
  Use when asked to "review prompt", "check persona", or "audit context assembly".
  Proactively use after changes to system_prompt.py, context_builder.py, or
  summarizer.py — any modification to what the LLM sees should trigger a prompt review.
context: fork
disable-model-invocation: true
---

# Prompt Review

Review the bot's system prompt and context assembly pipeline. Do NOT modify code — produce a structured report only.

## Arguments

$ARGUMENTS = optional focus area (e.g., "persona consistency", "token budget", "injection defense"). If empty, review all areas.

## Steps

1. Read the following files:
   - `src/bot/services/system_prompt.py` — persona definition
   - `src/bot/services/context_builder.py` — message assembly, token budgeting, memory merging
   - `src/bot/services/summarizer.py` — summary generation prompts
   - `src/bot/handlers.py` — how context flows from handler to LLM
2. Evaluate each area and rate 1-10:

   **A. Persona quality (system prompt)**
   - Is the Russian natural and consistent (not machine-translated)?
   - Are instructions clear, non-contradictory?
   - Is the persona specific enough to produce distinctive responses?
   - Are there missing behavioral boundaries?

   **B. Context assembly efficiency**
   - Is the token budget well-allocated between system prompt, history, memories, and user query?
   - Are there redundant or wasteful sections?
   - Is pruning logic correct (what gets dropped first when over budget)?
   - Are memories properly formatted for the LLM to use?

   **C. Prompt injection defense**
   - Are user messages and Cognee search results properly delimited?
   - Could a malicious user override system instructions via crafted messages?
   - Are memory facts sanitized before injection into prompt?

   **D. Summary/memory extraction prompts**
   - Does the summarizer extract the right level of detail?
   - Are extraction prompts biased toward useful vs. noisy facts?

3. Output a structured report:

```
## Prompt Review Report

### A. Persona Quality: X/10
- Finding 1
- Finding 2

### B. Context Assembly: X/10
...

### C. Injection Defense: X/10
...

### D. Memory Extraction: X/10
...

## Top 3 Improvements (by impact)
1. ...
2. ...
3. ...
```

## Gotchas

- Suggesting English fallback text — the bot persona is Russian-only; all user-facing strings must be in Russian.
- Recommending sync LangChain calls — must use `ainvoke`/`astream`, never `invoke`/`stream`.
- Ignoring token budget impact of suggestions — adding prompt content without accounting for the token budget breaks context assembly.
- Not checking memory/fact delimiter safety for injection — user-supplied memories can contain delimiter strings that break prompt structure.
- Evaluating persona quality with English-language standards — naturalness must be judged against native Russian, not translated English.
