---
paths:
  - "src/bot/handlers.py"
  - "src/bot/app.py"
  - "src/bot/adapters/**"
  - "src/bot/middlewares/**"
---

# aiogram 3.x patterns

This project uses aiogram 3.x with Router-based handlers and long-polling.

## Architecture
- Handlers in `src/bot/handlers.py` on a module-level `router = Router()`
- `app.py` creates `Dispatcher`, includes router, starts polling
- Bot: `Bot(token=..., parse_mode=ParseMode.HTML)` — replies use HTML tags
- Services injected via `dp.workflow_data` kwargs, not aiogram's DI

## Handler conventions
- All handlers `async def` with `-> None`
- `@router.message(CommandStart())` for `/start`, `@router.message()` as catch-all
- Access user: `message.from_user.id if message.from_user else 0`
- Reply: `await message.answer(text)`, not `await bot.send_message()`

<important>
- NEVER use aiogram 2.x: no `executor.start_polling()`, no `@dp.message_handler()`, no `Dispatcher(bot)`
- `CommandStart()` is a class instance (with parens), not a string filter
- Catch-all `@router.message()` with no filter MUST be registered LAST
- Always try/except with Russian fallback text in handlers
</important>

## Filters
- `from aiogram.filters import CommandStart, Command`
- `F` magic filter: `F.text`, `F.photo`, `F.chat.type == "private"`
- Combine: `&`, `|`, `~`; chain: `.regexp()`, `.startswith()`, `.in_()`

## FSM (if needed)
- `class MyStates(StatesGroup)` with `State()` fields
- Inject `state: FSMContext` — aiogram DI provides it
- Filter: `@router.message(MyStates.waiting_for_input)`
- Always `await state.clear()` on flow completion

## Middleware
- Subclass `BaseMiddleware`, implement `async def __call__(self, handler, event, data)`
- Register: `router.message.middleware(MyMiddleware())`

## CallbackData (inline keyboards)
- Subclass `CallbackData` with `prefix="..."` and typed fields
- Filter: `@router.callback_query(MyCallback.filter(F.action == "buy"))`
