# MEMORY_DESIGN — “Human-like” Memory for the Telegram Bot

**Status:** implemented (original design draft, updated to reflect Cognee migration from mem0)
**Scope:** memory only (text + attachments). Payments/subscriptions are out of scope except for metadata/limits.

## Goals
- **Stable current context**: the bot stays coherent about “what we’re doing now”.
- **Natural long-term recall**: the bot remembers important facts from the past without manual commands.
- **Automatic episode/session management**: new “episodes” start when topic/time changes.
- **Token-efficient**: avoid context rot; keep prompts compact.
- **Attachment-aware**: images/files become part of memory via text surrogates.

## Non-goals (for this phase)
- Multi-user group chat policy, privacy/GDPR workflows, multilingual strategy, persona/style persistence, safety policy enforcement.

---

## Conceptual Model
We implement **four product layers**, backed by **Cognee** for knowledge-graph memory.

### Product Layers (canonical in our app)
1) **Working memory (short-term)**
   - last *N* messages (sliding window)
   - `running_summary`
   - `open_loops` / current goals

2) **Episode memory (session/episodic)**
   - raw message log for the episode
   - `chunk_summaries` (periodic)
   - `final_episode_summary` on close

3) **Semantic long-term memory**
   - durable “memory units” (facts, preferences, decisions)
   - vector search + knowledge graph via Cognee

4) **Artifacts (attachments)**
   - stored as objects + derived text representations (vision/OCR/transcripts/summaries/chunks)
   - searchable independently

### Mapping to Cognee
Cognee provides: **chunk storage** (via `cognee.add()`) + **knowledge graph** (via `cognee.cognify()`) + **vector search** (via `cognee.search()`).

- **Our working memory** stays in our DB (fast, deterministic). Not stored in Cognee.
- **Our episode summaries** and conversation pairs are written to Cognee datasets (per-user: `tg_user_{user_id}`).
- **Our durable memory units** are stored as raw text in Cognee; `cognify()` builds a knowledge graph from them for richer retrieval.
- **Artifacts** remain in our DB+storage; only high-level conclusions may be written to Cognee.

---

## Identifiers & Scoping
We standardize ids for consistent retrieval.

- `user_id` = Telegram user id (stable)
- `thread_id` = one conversational stream per user/chat (stable)
- `episode_id` = one “chapter” within thread (changes over time)
- `dataset_name` (Cognee) = `tg_user_{user_id}` (per-user isolation)

**Rule:** All Cognee writes MUST include `dataset_name=f"tg_user_{user_id}"` for user isolation.

---

## Data Model (Supabase Postgres)
> Exact SQL/migrations can be added later; this section defines the target schema.

### 1) Threads & Episodes
- `threads`
  - `id` (uuid)
  - `telegram_user_id` (bigint)
  - `active_episode_id` (uuid, nullable)
  - timestamps

- `episodes`
  - `id` (uuid)
  - `thread_id` (uuid)
  - `status` = `active|closed`
  - `started_at`, `ended_at`
  - `topic_label` (text, optional)
  - `last_user_message_at`

### 2) Messages (raw log)
- `messages`
  - `id` (uuid)
  - `episode_id` (uuid)
  - `role` = `user|assistant|system`
  - `content_text` (text)
  - `created_at`
  - optional: `tokens_in`, `tokens_out`, `model`

### 3) Episode Summaries
- `episode_summaries`
  - `id` (uuid)
  - `episode_id` (uuid)
  - `kind` = `running|chunk|final`
  - `summary_text` (text) — short, LLM-friendly
  - `summary_json` (jsonb) — structured fields (see below)
  - `created_at`

**Structured format (`summary_json`)**
```json
{
  "topic": "...",
  "decisions": ["..."],
  "todos": ["..."],
  "facts_candidates": [{"text":"...","confidence":0.8}],
  "entities": ["..."],
  "artifacts": [{"artifact_id":"...","note":"..."}],
  "open_questions": ["..."]
}
```

### 4) Artifacts (attachments)
- `artifacts`
  - `id` (uuid)
  - `episode_id` (uuid)
  - `message_id` (uuid)
  - `type` = `image|file|audio|video|link`
  - `storage_url` (text)
  - `mime` (text)
  - `size_bytes` (bigint)
  - `sha256` (text) — dedupe
  - `created_at`

- `artifact_text`
  - `id` (uuid)
  - `artifact_id` (uuid)
  - `text_kind` = `user_caption|vision_short|vision_detail|ocr|file_summary|chunk|transcript`
  - `text` (text)
  - `embedding` (vector) — pgvector
  - `confidence` (float, nullable)
  - `created_at`

---

## Episode Lifecycle (automatic, no user commands)
### Start a new episode when ANY of these triggers fires
1) **Time gap**: `now - last_user_message_at > TIME_GAP_HOURS` (e.g., 8–12h)
2) **Topic shift**: cosine similarity between new message and `running_summary` embedding < `TOPIC_SIM_THRESHOLD` (e.g., 0.70)
3) **Explicit markers**: user says “давай про другое”, “кстати”, etc.

### Anti-flap / hysteresis
- Topic shift must persist for **2 consecutive user turns** OR be combined with explicit markers.

### Close episode
On close:
- generate **final summary**
- extract **memory units** (facts to remember)
- write to Cognee (knowledge graph)

---

## Summarization Strategy
### 1) Running summary (cheap, frequent)
Update when:
- every `M` messages (e.g., 10–15) OR
- token pressure is high

Output:
- `summary_text` (1–2 short paragraphs)
- `open_loops` list (goals/todos)

### 2) Chunk summaries (optional)
For long episodes:
- every 30–50 messages, store a `chunk` summary

### 3) Final episode summary
On episode close:
- produce structured JSON + short text
- extract `facts_candidates` for long-term memory

**Extraction model settings:** temperature ≤ 0.2.

---

## Cognee Integration (Knowledge Graph + Vector Search)
We use Cognee as a **long-term memory engine** with knowledge graph capabilities.

### What Cognee provides
- Chunk storage via `cognee.add()` — fast, stores raw text
- Knowledge graph construction via `cognee.cognify()` — builds entity/relationship graph
- Vector similarity search via `cognee.search()` — retrieves relevant chunks

### What we implement ourselves
- Episode management (Telegram-specific)
- Attachments pipeline + artifact index
- TTL/decay maintenance (`MemoryCleanupService`)
- User isolation via per-user datasets (`tg_user_{user_id}`)

### Write policy (selective)
We do **not** store every message.

We store:
- conversation pairs (user message + bot response) via `write_factual()`
- `cognify()` runs periodically (every 10 writes) to build the knowledge graph

We ignore (at cognee level):
- greetings, filler (handled by future fact extraction layer)
- system messages

### Implementation details
- `CogneeMemoryService` wraps all Cognee operations
- `cognify()` uses a single lock to prevent concurrent graph builds
- Search returns `MemoryFact` objects with `MemoryType` and `MemoryCategory` enums
- Raw search results are converted to typed data models at the service boundary

---

## Context Building (per response)
We follow a **dual-memory read flow** (production pattern):

### Inputs
- new user message
- current episode state (running summary, recent messages)

### Steps
1) Load **working memory** from DB
   - `running_summary`
   - last `N` messages
   - short artifact surrogates
2) In parallel, query Cognee:
   - `cognee_service.search(query, user_id, limit=5)`
   - searches within user's dataset (`tg_user_{user_id}`)
3) If user asks about an attachment:
   - retrieve from `artifact_text` index (pgvector)
4) Assemble prompt:
   - system/persona
   - running summary + open loops
   - recent messages
   - top-K Cognee memories (compressed)
   - relevant artifact snippets
5) Prune duplicates and resolve conflicts

**Optional:** enable reranking for precision-critical queries (extra latency).

---

## Attachment Handling
### Images
- store the file
- generate:
  - `vision_short` (1–2 lines)
  - `vision_detail` (paragraph)
  - `ocr` (if text present)
- insert a text surrogate into the message history:
  - `[image#A123] краткое описание…`

### Documents/files
- extract text (if possible)
- chunk by structure
- store `file_summary` + chunk embeddings in `artifact_text`

### What goes to Cognee
Only high-level conclusions, e.g.
- “User sent a screenshot of ImportError: create_client…”

---

## Validation Checklist
A build is considered correct when:
- Episode switching is automatic and stable (no flapping).
- Prompts stay within budget via summaries + top-K retrieval.
- Long-term recall works for durable facts/decisions.
- Attachments are retrievable and referenced via text surrogates.
- Cognee storage does not bloat (selective write + TTL/decay maintenance).

---

## Implementation Status

All core memory features are implemented:
- Episode management with auto-switching (time gap + topic shift) — `conversation/episode_manager.py`
- Running/chunk/final summarization — `conversation/summarizer.py`
- Cognee-backed semantic memory — `memory/cognee_service.py`
- TTL/decay cleanup — `memory/cleanup.py`
- Artifact pipeline — `media/artifact_service.py`
- Dual-memory context building — `conversation/context_builder.py`
