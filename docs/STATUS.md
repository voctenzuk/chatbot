# Project Status — voctenzuk/chatbot

Last updated: 2026-02-04

## ✅ Recently merged (today)
- PR #9 — ContextBuilder dual-memory prompt assembly
  https://github.com/voctenzuk/chatbot/pull/9
- PR #10 — Memory DB schema (threads/episodes/messages/episode_summaries) + tests
  https://github.com/voctenzuk/chatbot/pull/10
- PR #11 — Auto episode switching (topic/time + anti-flap)
  https://github.com/voctenzuk/chatbot/pull/11
- PR #12 — mem0 integration wrapper (OSS + pgvector)
  https://github.com/voctenzuk/chatbot/pull/12
- PR #13 — PR template + Codex/GitHub workflow hygiene
  https://github.com/voctenzuk/chatbot/pull/13

## 🧠 Memory system (current state)
We now have:
- Canonical DB schema for **threads/episodes/messages/summaries**
- ContextBuilder (working window + summary + mem0 retrieval)
- mem0 wrapper (write factual/episodic + search)
- Auto episode switching logic
- CI + formatting + Codex review workflow

## 🔜 Next work (open epics)
Open GitHub issues:
- #2 — EpisodeManager service
- #3 — Summarizer (running + final)
- #6 — Artifacts pipeline (attachments)
- #7 — Auto episode switching (**already merged in #11** → should be closed or repurposed)
- #8 — TTL/decay maintenance

## Recommended order (autonomous default)
1) **#2 EpisodeManager** — needed to bind every message to an episode and provide a stable API for the rest.
2) **#3 Summarizer** — running + final summaries; produces memory-unit candidates for mem0.
3) **#6 Artifacts** — attachments storage + text surrogates + retrieval.
4) **#8 TTL/decay** — cleanup/maintenance to prevent bloat.

## Acceptance criteria per epic (definition of done)
### #2 EpisodeManager
- API exists and is used by handlers so every message has an `episode_id`.
- Episode start/close works (time-gap + explicit close) and is unit-tested.

### #3 Summarizer
- Running summary updates (every M messages or token pressure).
- Final summary JSON matches `ARCHITECTURE/MEMORY_DESIGN.md`.
- Extracts `facts_candidates` for mem0; unit-tested.

### #6 Artifacts
- Tables + storage for artifacts; sha256 dedupe.
- Images: vision_short/detail + OCR saved as text; retrievable.
- Docs: extract→chunk→embed + file_summary.
- ContextBuilder can include artifact surrogates when relevant.

### #8 TTL/decay
- Importance heuristic + TTL fields/metadata.
- Scheduled cleanup job; documented tuning knobs.

## Notes for new session (/new)
If you start a fresh chat and say “continue chatbot work”, use this file + the open issues list as the source of truth.
