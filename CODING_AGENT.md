# Coding agent (Kimi K2.5) — workflow

## What this is
A repeatable way to ask a **coding-focused agent** (model: **Kimi K2.5**) to implement tasks in this repo.

## How to use (when working with Molter/OpenClaw)
When you want code changes, write your request like:
- Goal (what feature)
- Constraints (libs: aiogram, LangChain/LangGraph, mem0)
- Acceptance criteria (what must work)
- Files/structure preferences

And explicitly say: **“Use Kimi K2.5 for coding”**.

## Coding standards for this repo
- Python 3.12, async-first
- Strict per-user isolation: all keys based on Telegram `from_user.id`
- No long-term memory writes without explicit user content (facts/preferences)
- Images: generate only on explicit request (photo/selfie)
- Logging: structured, no secrets

## Typical tasks to delegate
- Add new handlers and conversation flows
- Implement LangGraph state machine and tool-calling
- Integrate mem0 (add/search) with per-user namespace
- Add image generation route + safe prompt builder
