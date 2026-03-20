---
name: add-handler
description: >
  Scaffold a new Telegram command, message handler, or callback query handler.
  Use when asked to "add command", "new handler", "bot command", or "callback handler".
  Proactively use when the user describes a new bot interaction (e.g. "users should be
  able to /settings" or "add inline keyboard for X") — scaffold the handler before
  implementing business logic.
disable-model-invocation: true
---

# Add Handler

Add a new Telegram bot handler following project conventions.

## Arguments

$ARGUMENTS = handler type and purpose (e.g., "/help command" or "photo message handler" or "callback for inline keyboard")

## Steps

1. Read `src/bot/handlers.py` to understand current handler patterns, imports, and registration order.
2. Determine handler type from arguments:
   - **Command** (`/start`, `/help`, etc.) → `@router.message(Command("name"))`
   - **CommandStart** → `@router.message(CommandStart())`
   - **Message filter** (photo, text, etc.) → `@router.message(F.photo)` or content type filter
   - **Callback query** → `@router.callback_query(MyCallback.filter(...))`
   - **Catch-all** → `@router.message()` with NO filter — MUST be registered last
3. Write the handler in `src/bot/handlers.py` following these conventions:
   - `async def handler_name(message: Message) -> None:` signature
   - Defensive user access: `user_id = message.from_user.id if message.from_user else 0`
   - Try/except wrapping service calls
   - Russian fallback text on error: `await message.answer("Прости, ...")`
   - Loguru logging: `logger.error("Description {}", exc)`
   - Use `await message.answer()`, not `await bot.send_message()`
4. If the handler needs new service imports, add them following the existing pattern (try/except for optional services like cognee, db_client).
5. **Critical**: verify the catch-all `@router.message()` with no filter remains LAST. Insert new handlers ABOVE it.
6. If the handler introduces a new command, note that it should be registered with BotFather.
7. Run `uvx ruff check --fix src/bot/handlers.py && uvx ruff format src/bot/handlers.py`.
8. Print a summary of what was added and where.

## Gotchas

- Placing new handler AFTER the catch-all `@router.message()` — it will never fire because the catch-all matches first. Always insert above it.
- Using `bot.send_message()` instead of `message.answer()` — handlers should reply via the message object.
- Forgetting try/except with Russian fallback text — every handler must catch service exceptions and reply in Russian.
- Using aiogram 2.x patterns (`dp.message_handler`, `executor`) — this project uses aiogram 3 with `Router` and `Dispatcher`.
- Not checking `message.from_user` for `None` before accessing `.id` — channel posts and anonymous messages have no `from_user`.
