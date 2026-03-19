---
name: add-migration
description: When creating or modifying database tables, columns, indexes, or RLS policies
disable-model-invocation: true
---

# Add Migration

Create a new SQL migration file for the Supabase database.

## Arguments

$ARGUMENTS = migration description (e.g., "add user preferences table")

## Steps

1. Look at existing SQL files in the project (`sql/` or root `*.sql`) to understand current patterns and naming conventions.
2. Create a timestamped migration file: `sql/YYYYMMDD_HHMMSS_<snake_case_description>.sql`
3. Structure the file with `-- migrate:up` and `-- migrate:down` sections.
4. Follow these rules:
   - Use `IF NOT EXISTS` for `CREATE TABLE/INDEX`, `IF EXISTS` for `DROP` — idempotency is required.
   - Add `created_at TIMESTAMPTZ DEFAULT now()` to all new tables.
   - Add Row Level Security (RLS) policies scoped to `user_id` for any user-facing table.
   - Use `BIGINT` for Telegram user IDs.
   - Reference existing table naming conventions (snake_case, plural).
5. Run `cat` on the created file to confirm contents.
6. Print the file path and a summary of the migration.

## Gotchas

- Forgetting `IF NOT EXISTS` / `IF EXISTS` for idempotency — every CREATE and DROP must be guarded.
- Using `INTEGER` instead of `BIGINT` for Telegram user IDs — Telegram IDs exceed 32-bit range.
- Missing RLS policies on user-facing tables — any table with a `user_id` column needs row-level security.
- Forgetting `created_at TIMESTAMPTZ DEFAULT now()` — all new tables must have this column.
- Writing a down migration that doesn't fully reverse the up migration — always verify the down section drops/reverts everything the up section creates.
