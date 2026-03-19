# Memory Task Breakdown (assignable tickets)

Use this as a checklist for assigning to sub-agents. Each task has clear inputs/outputs.

## A. Schema & migrations
- [ ] A1. Add migrations for `threads`, `episodes`, `messages`, `episode_summaries`
- [ ] A2. Add migrations for `artifacts`, `artifact_text` (pgvector)
- [ ] A3. Add indexes + constraints; verify with a SQL smoke test

## B. EpisodeManager
- [ ] B1. Implement EpisodeManager API (create/get thread, start/close episode)
- [ ] B2. Wire into handlers: ensure every message belongs to an episode
- [ ] B3. Tests for lifecycle and timestamps

## C. Summarizer
- [ ] C1. Running summary updater (trigger every M messages or token pressure)
- [ ] C2. Final summary generator on close (structured JSON)
- [ ] C3. Memory unit extractor from final summary

## D. mem0 integration
- [ ] D1. Add mem0 pgvector config (OSS)
- [ ] D2. Implement wrapper methods (write factual/episodic/search)
- [ ] D3. Add ingestion instructions (what to store/ignore)
- [ ] D4. Add tests/mocks for mem0 client

## E. ContextBuilder
- [ ] E1. Build context: system + running summary + last N messages
- [ ] E2. Add mem0.search in parallel, with metadata filters
- [ ] E3. Pruning rules, top-K limits, conflict handling

## F. Artifacts
- [ ] F1. Artifact upload + dedupe via sha256
- [ ] F2. Image pipeline: vision_short/detail + OCR, stored in artifact_text
- [ ] F3. Doc pipeline: extract→chunk→embed; file_summary
- [ ] F4. Audio pipeline: transcript + summary
- [ ] F5. Artifact retrieval (“that photo/file”) + context injection

## G. Auto episode switching
- [ ] G1. Implement time-gap trigger
- [ ] G2. Implement embedding-based topic shift detection
- [ ] G3. Anti-flap logic + tests

## H. Maintenance
- [ ] H1. Add importance scoring (heuristic)
- [ ] H2. TTL/cleanup cron job

## Definition of Done (DoD)
- Memory works across: working, episode, long-term (mem0), artifacts.
- No manual session switching for normal operation.
- Summaries keep prompts within budget.
- Selective write prevents memory bloat.
- Attachment recall works.
