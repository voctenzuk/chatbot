# MEMORY_DESIGN — “Human-like” Memory for the Telegram Bot

**Status:** design draft (implementation roadmap included)  
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
We implement **four product layers**, and map them onto **mem0’s memory types**.

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
   - vector search for recall

4) **Artifacts (attachments)**
   - stored as objects + derived text representations (vision/OCR/transcripts/summaries/chunks)
   - searchable independently

### Mapping to mem0 types
mem0 has: **Working / Factual / Episodic / Semantic**.

- **Our working memory** stays primarily in our DB (fast, deterministic). Optionally mirrored to mem0 with `run_id`.
- **Our episode summaries** map to **mem0 Episodic** (write summaries, not raw logs).
- **Our durable memory units** map to **mem0 Factual** (primary) and optionally Semantic.
- **Artifacts** remain in our DB+storage; only high-level conclusions are written to mem0.

---

## Identifiers & Scoping
We standardize ids for consistent retrieval.

- `user_id` = Telegram user id (stable)
- `thread_id` = one conversational stream per user/chat (stable)
- `episode_id` = one “chapter” within thread (changes over time)
- `run_id` (mem0) = **episode_id** (1:1 mapping)

**Rule:** All mem0 writes MUST include `user_id` and SHOULD include `run_id`.

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
- write to mem0 (factual + episodic)

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
- extract `facts_candidates` for mem0

**Extraction model settings:** temperature ≤ 0.2.

---

## mem0 Integration (OSS + pgvector)
We use mem0 as a **long-term memory engine**.

### What we delegate to mem0
- fact extraction + dedup + conflict resolution (**`infer=True`**)
- semantic search + memory merging
- metadata filtering

### What we implement ourselves
- episode management (Telegram-specific)
- attachments pipeline + artifact index
- TTL/decay (mem0 does not provide this out of the box)

### Write policy (selective)
We do **not** store every message.

We store:
- user preferences, stable facts
- confirmed decisions / agreements
- important episodic summaries (final summary)

We ignore:
- greetings, filler
- speculative/uncertain info
- transient chatter

### Recommended mem0 usage
- **Factual**: write extracted “memory units” with metadata.
- **Episodic**: write final episode summary with `run_id`.
- **Working**: keep in DB; optional mirror to mem0 if needed.

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
2) In parallel, query mem0:
   - `mem0.search(user_id, query, filters)`
   - filters include `project`, optionally `thread_id`
3) If user asks about an attachment:
   - retrieve from `artifact_text` index (pgvector)
4) Assemble prompt:
   - system/persona
   - running summary + open loops
   - recent messages
   - top-K mem0 memories (compressed)
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

### What goes to mem0
Only high-level conclusions, e.g.
- “User sent a screenshot of ImportError: create_client…”

---

## Validation Checklist
A build is considered correct when:
- Episode switching is automatic and stable (no flapping).
- Prompts stay within budget via summaries + top-K retrieval.
- Long-term recall works for durable facts/decisions.
- Attachments are retrievable and referenced via text surrogates.
- mem0 storage does not bloat (selective write + future TTL).

---

## Implementation Roadmap (high level)
See `docs/roadmap/MEMORY_ROADMAP.md` and tasks in `docs/tasks/memory/*`.
