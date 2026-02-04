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
- #7 — Auto episode switching (already merged in #11, issue can be closed/updated)
- #8 — TTL/decay maintenance

## Notes for new session (/new)
If you start a fresh chat and say “continue chatbot work”, use this file + the open issues list as the source of truth.
