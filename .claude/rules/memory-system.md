---
paths:
  - "src/bot/services/cognee_memory_service.py"
  - "src/bot/services/episode_manager.py"
  - "src/bot/services/episode_switcher.py"
  - "src/bot/services/context_builder.py"
  - "src/bot/services/memory_*.py"
---

# Memory system (Cognee + episodes)

## Dual-layer architecture
- **Working memory** (Postgres/Supabase): last N messages, running_summary, open_loops. Fast, deterministic.
- **Semantic memory** (Cognee knowledge graph): durable facts, preferences, episode summaries. Vector + graph search.

## Episode model
- Episodes auto-switch on time gap (8h default) or topic shift (cosine sim < 0.7).
- `episode_id` maps 1:1 to `run_id`. All writes carry `user_id` + `run_id`.
- On episode close: generate final summary, extract memory units, write to Cognee.

## Cognee API pipeline (all async)
```
await cognee.add(data, dataset_name)   # 1. Ingest text (fast)
await cognee.cognify()                  # 2. Build knowledge graph (heavy)
results = await cognee.search(          # 3. Query
    query_text=q,
    query_type=SearchType.CHUNKS        #    or GRAPH_COMPLETION
)
```

## Key patterns in this project
- User isolation via per-user datasets: `tg_user_{user_id}`.
- `CogneeMemoryService` wraps cognee behind `CogneeClientProtocol` for testability.
- `_pending_datasets` tracks datasets needing cognify; cleared after `cognify()`.
- `ContextBuilder.assemble()` merges summary + recent messages + semantic memories + artifact surrogates, pruned to token budget.
- Search results are normalized to `MemoryFact` via `_extract_text()` (handles str, dict, objects with `search_result`).

## Selective write policy
Store: user preferences, stable facts, confirmed decisions, final episode summaries.
Skip: greetings, filler, speculative info, transient chatter.

<important>
- All Cognee calls are async. Always `await` add/cognify/search — forgetting causes silent data loss.
- Never call `cognify()` after every single write. It is expensive. Batch writes, then cognify periodically or on episode close.
- Always scope data to `tg_user_{user_id}` dataset. Omitting dataset_name breaks user isolation.
- `prune_data()` is global — it wipes ALL data, not just one user. Use with extreme caution.
- Config (LLM key, model, endpoint) must be set before first add/cognify call.
</important>
