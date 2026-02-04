# MEMORY_ROADMAP — Implementation Plan

This roadmap decomposes memory into parallelizable work packages.  
Target stack: **aiogram 3.x + Supabase Postgres + pgvector + mem0 (OSS)**.

## Milestone 0 — Baseline / prerequisites (0.5–1 day)
- [ ] Ensure repo installs/tests in CI (already in progress)
- [ ] Confirm Postgres schema/migrations workflow (Supabase migrations directory naming finalized)
- [ ] Decide storage for artifacts: **Supabase Storage** (default)

Deliverable:
- developer can run bot + tests locally

---

## Milestone 1 — Canonical episode storage (DB) (1–2 days)
**Goal:** Persist threads/episodes/messages/summaries.

Deliverables:
- migrations for `threads`, `episodes`, `messages`, `episode_summaries`
- EpisodeManager service
- unit tests for episode lifecycle

Parallel tasks:
- schema/migrations
- service + tests

---

## Milestone 2 — Summarization v1 (1–2 days)
**Goal:** Running summary + final summary.

Deliverables:
- Summarizer service (running + final)
- structured JSON summary format
- integration in message handling flow

---

## Milestone 3 — mem0 integration (factual + episodic) (1–2 days)
**Goal:** Use mem0 for long-term memory using pgvector.

Deliverables:
- mem0 config for pgvector
- MemoryService wrapper:
  - `write_factual(user_id, text, metadata)`
  - `write_episodic(user_id, run_id, summary, metadata)`
  - `search(user_id, query, filters)`
- project-level ingestion instructions (selective memory)

---

## Milestone 4 — ContextBuilder v1 (dual-memory) (1–2 days)
**Goal:** Build prompts from working memory + mem0 search.

Deliverables:
- ContextBuilder service
- metadata filtering strategy
- pruning + top-K limits

---

## Milestone 5 — Artifacts pipeline (2–4 days)
**Goal:** Attachments become searchable context.

Deliverables:
- Artifacts storage (upload + dedupe)
- artifact_text generation:
  - images: vision_short/detail + OCR
  - docs: extract + chunk + summary
  - audio: transcript + summary
- artifact retrieval for “that file/photo” queries

---

## Milestone 6 — Auto episode switching (topic/time) (1–3 days)
**Goal:** Natural session/episode boundaries.

Deliverables:
- embedding-based topic shift detection
- time-gap trigger
- anti-flap rules
- tests with synthetic conversations

---

## Milestone 7 — Decay/TTL + maintenance jobs (1 day)
**Goal:** Prevent memory bloat.

Deliverables:
- importance scoring (simple heuristic)
- TTL/cleanup job (cron)
- metrics/logging

---

## Acceptance Criteria (Definition of Done)
- Memory works across 4 layers: working, episode, long-term, artifacts.
- No manual session switching commands required for normal use.
- Summaries keep context under budget.
- mem0 stores selective, high-value memories and can recall them.
- Attachments are represented in context via text surrogates and searchable.
