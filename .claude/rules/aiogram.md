---
paths:
  - "src/bot/handlers.py"
  - "src/bot/app.py"
  - "src/bot/middlewares/**"
---

# aiogram 3.x patterns

This project uses aiogram 3.x with Router-based handler organization and long-polling (`dp.start_polling(bot)`).

## Architecture

- Handlers live in `src/bot/handlers.py` on a module-level `router = Router()`.
- `app.py` creates `Dispatcher`, includes the router via `dp.include_router(router)`, and starts polling.
- Bot is instantiated with `Bot(token=..., parse_mode=ParseMode.HTML)` -- all replies can use HTML tags.
- Services are accessed via module-level getter functions (e.g., `get_llm_service()`, `get_episode_manager()`), not aiogram's DI.

## Handler conventions

- All handlers are `async def` with return type `-> None`.
- Use `@router.message(CommandStart())` for `/start`, `@router.message()` as the catch-all.
- Access user info defensively: `message.from_user.id if message.from_user else 0`.
- Reply with `await message.answer(text)`, not `await bot.send_message()`.

## Filters

- Import filters from `aiogram.filters` (e.g., `CommandStart`, `Command`).
- Use `F` magic filter for content/attribute filtering: `F.text`, `F.photo`, `F.chat.type == "private"`.
- Combine with `&`, `|`, `~` operators; chain with `.regexp()`, `.startswith()`, `.in_()`.

<important>
- NEVER use aiogram 2.x patterns: no `executor.start_polling()`, no `@dp.message_handler()`, no `Dispatcher(bot)`.
- In 3.x: `@router.message()` replaces `@dp.message_handler()`. Dispatcher takes NO bot argument.
- `CommandStart()` is a class instance (with parens), not a string filter.
- Catch-all `@router.message()` with no filter MUST be registered last -- it matches everything.
</important>

## FSM (if needed)

- Define states as `class MyStates(StatesGroup)` with `State()` fields.
- Inject `state: FSMContext` in handler signature -- aiogram DI provides it automatically.
- Filter by state: `@router.message(MyStates.waiting_for_input)`.
- Always call `await state.clear()` when the flow completes.

## Middleware

- Subclass `BaseMiddleware`, implement `async def __call__(self, handler, event, data)`.
- Register: `router.message.middleware(MyMiddleware())`.
- Outer middleware (runs before filters): `router.message.outer_middleware(...)`.

## Error handling

- Use `@router.errors()` for centralized error handling across handlers.
- Current pattern: try/except inside each handler with `loguru` logger and user-facing fallback messages in Russian.

## CallbackData (inline keyboards)

- Subclass `CallbackData` with `prefix="..."` and typed fields.
- Filter: `@router.callback_query(MyCallback.filter(F.action == "buy"))`.
- Handler receives `callback_data: MyCallback` via DI.
