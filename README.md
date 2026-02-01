# Telegram “Virtual Girlfriend” Bot (aiogram + LangChain/LangGraph + mem0)

This repository is currently a scaffold.

## Goals
- Telegram chat bot (Python 3.12, **aiogram**)
- “Alive” conversational behavior (persona + consistency + context retention)
- **Short-term memory** (chat context per user) + **long-term memory** (facts/preferences) via **mem0**
- Optional **image generation** when user asks for “photos/selfies”

## Quick start (local)
1) Create `.env` from `.env.example`
2) Start infra:
```bash
docker compose up -d
```
3) Install deps (example with uv):
```bash
uv venv
uv pip install -e .
```
4) Run:
```bash
python -m bot
```

## Notes
- Keep per-user isolation strict (Telegram `from_user.id` → memory/session key).
- Images should be generated only when explicitly requested.
