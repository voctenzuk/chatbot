# TODOS

## P1 — Critical

### Unit Economics Validation
**What:** Посчитать реальную цену kimi-k2p5 за токен, настроить cost_cents трекинг, пересмотреть прайсинг по реальным данным.
**Why:** При оценочных ценах даже Pro тиер ($9.99) может быть убыточным (~$36 себестоимость при 100 сообщ/день). Без реальных данных невозможно установить правильные цены.
**Context:** `usage_tracking` уже трекает `tokens_input` и `tokens_output`. Нужно добавить `cost_cents` (принято в scope CEO review) и заполнять его реальными ценами API. Затем проанализировать данные и скорректировать тиеры/лимиты.
**Effort:** S (CC ~15 min для трекинга, затем ручной анализ данных)
**Depends on:** cost_cents tracking (в scope текущего плана)

## P2 — Important

### /stats Command — Usage Dashboard
**What:** Команда /stats показывает пользователю: сообщений сегодня (15/20), план (Free), дней вместе (12).
**Why:** Делает лимит прозрачным, мотивирует апгрейд на платный тиер. Пользователь видит ценность и понимает что получит при апгрейде.
**Context:** Данные уже есть в `usage_tracking` и `user_subscriptions`. Нужен handler для /stats + запрос к `get_user_usage_today()` SQL функции.
**Effort:** S (CC ~15 min)
**Depends on:** rate limiting и usage tracking (в scope текущего плана)

### Relationship Levels (Gamification)
**What:** Использовать `relationship_depth` (0-10) из `memory_models.py`. По мере общения уровень растёт → бот становится теплее, отправляет больше фото, пишет чаще первым.
**Why:** Retention mechanism — пользователь не хочет терять прогресс. Создаёт ощущение развивающихся отношений.
**Context:** Поле `relationship_depth` уже существует в `MemoryFact`. Нужно: логика увеличения уровня, влияние уровня на system prompt и proactive messaging частоту.
**Effort:** M (CC ~30 min)
**Depends on:** работающая память (memory writing) и проактивные сообщения

## P3 — Nice to Have

### Scaling Path Documentation
**What:** Документ с конкретными триггерами масштабирования: 100+ DAU → webhooks, 500+ DAU → отдельный proactive worker, 1000+ DAU → cognee worker.
**Why:** Текущая архитектура (polling, single process) работает на ~100 пользователей. Нужно знать заранее, когда и что менять.
**Context:** Polling mode в aiogram, proactive scheduler в том же процессе, cognee in-process. Каждый из этих компонентов — потенциальное bottleneck.
**Effort:** S (CC ~15 min, это документ, не код)
**Depends on:** ничего
