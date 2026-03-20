---
paths:
  - "src/bot/memory/**"
  - "src/bot/conversation/episode_manager.py"
  - "src/bot/conversation/episode_switcher.py"
  - "src/bot/conversation/context_builder.py"
---

# Memory system (Cognee + episodes)

## Dual-layer architecture
- **Working memory** (Postgres/Supabase): last N messages, running_summary, open_loops
- **Semantic memory** (Cognee knowledge graph): durable facts, preferences, episode summaries

## Episode model
- Auto-switch on time gap (8h default) or topic shift (cosine sim < 0.7)
- `episode_id` maps 1:1 to `run_id`. All writes carry `user_id` + `run_id`
- On close: generate summary, extract memory units, write to Cognee

## Cognee API pipeline (all async)
```
await cognee.add(data, dataset_name)   # 1. Ingest (fast)
await cognee.cognify()                  # 2. Build KG (heavy)
results = await cognee.search(query)    # 3. Query
```

<important>
- All Cognee calls are async. Always `await` — forgetting causes silent data loss
- Never `cognify()` after every write. Batch, then cognify on episode close
- Always scope to `dataset_name=f"tg_user_{user_id}"`. Omitting breaks user isolation
- `prune_data()` is GLOBAL — wipes ALL data, not one user. Never call it
- Config (LLM key, model, endpoint) must be set before first add/cognify
</important>

## Key patterns
- `CogneeMemoryService` wraps cognee behind `CogneeClientProtocol` for testability
- `ContextBuilder.assemble()` merges summary + recent messages + semantic memories, pruned to token budget
- Search results normalized to `MemoryFact` via `_extract_text()`

## Selective write policy
Store: user preferences, stable facts, confirmed decisions, episode summaries.
Skip: greetings, filler, speculative info, transient chatter.
