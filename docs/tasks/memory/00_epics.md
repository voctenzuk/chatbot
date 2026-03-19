# Memory Epics (work packages)

Each epic is designed so a developer/sub-agent can work independently with minimal overlap.

## EPIC A — DB schema: threads/episodes/messages/summaries
Owner: schema/dev
- Create migrations for:
  - `threads`, `episodes`, `messages`, `episode_summaries`
- Add indexes on (telegram_user_id), (episode_id, created_at)
- Ensure RLS strategy aligns with existing Supabase patterns

Acceptance:
- migrations apply cleanly
- unit test or smoke SQL verifies tables

---

## EPIC B — EpisodeManager service
Owner: backend/dev
- Implement:
  - `get_or_create_thread(user_id)`
  - `get_active_episode(thread_id)`
  - `start_episode(thread_id)`
  - `close_episode(episode_id, reason)`
  - update `last_user_message_at`
- Add tests for episode lifecycle

Acceptance:
- tests cover time-gap close and explicit close

---

## EPIC C — Summarizer (running + final)
Owner: backend/dev
- Implement running summary updates
- Implement final structured summary on close
- Store into `episode_summaries`
- Keep extraction temperature ≤ 0.2

Acceptance:
- summary format matches MEMORY_DESIGN

---

## EPIC D — mem0 integration wrapper
Owner: backend/dev
- Implement `Mem0MemoryService` wrapper with:
  - `write_factual(user_id, text, metadata, infer=True)`
  - `write_episodic(user_id, run_id, text_or_json, metadata, infer=True)`
  - `search(user_id, query, filters, k)`
- Ensure pgvector provider config
- Add “ingestion instructions” text (what to store/ignore)

Acceptance:
- can write and retrieve memories in a local/dev environment

---

## EPIC E — ContextBuilder (dual-memory)
Owner: backend/dev
- Build prompt context from:
  - running summary
  - last N messages
  - mem0 search results (top-K)
  - artifact surrogates (later)
- Add pruning/dedupe and conflict handling (prefer recent)

Acceptance:
- deterministic ordering and size limits

---

## EPIC F — Artifacts pipeline
Owner: backend/dev
- Create migrations for `artifacts` + `artifact_text`
- Implement upload/dedupe and store metadata
- Implement derived texts:
  - images: vision short/detail + OCR
  - docs: extract + chunk + summary
  - audio: transcript + summary
- Add artifact retrieval and integration into context builder

Acceptance:
- “what was in that photo/file?” works via retrieval

---

## EPIC G — Auto episode switching
Owner: backend/dev
- Implement topic shift detection (embeddings + threshold)
- Anti-flap rules
- Time-gap trigger

Acceptance:
- tests with synthetic dialogues show stable switching

---

## EPIC H — TTL/decay maintenance
Owner: backend/dev
- Add importance score heuristic
- Add cleanup job (cron) for expiring memories

Acceptance:
- memory store does not grow unbounded in long runs
