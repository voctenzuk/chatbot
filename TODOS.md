# TODOS

## P1 — Critical

### Unit Economics Validation
**What:** Проверить реальную цену kimi-k2p5 за токен против констант COST_PER_1M_INPUT/OUTPUT в `chat_pipeline.py`. После 1 недели данных — проанализировать cost_cents и скорректировать тиеры/лимиты.
**Why:** При оценочных ценах даже Pro тиер ($9.99) может быть убыточным. Нужны реальные данные для правильного прайсинга.
**Context:** `cost_cents NUMERIC(10,4)` уже трекается в `usage_tracking` (миграция 008). Константы: `COST_PER_1M_INPUT=0.15`, `COST_PER_1M_OUTPUT=0.60`. Image cost теперь приходит из `ImageResult.cost_cents` (по провайдеру). После накопления данных — сравнить с реальными API счетами (OpenRouter + kimi-k2p5).
**Effort:** S (ручной анализ данных после 1 недели)
**Depends on:** Character MVP (завершён)

## P2 — Important

### OpenRouter Image Cost Monitoring
**What:** После 1 недели с FLUX/SeeDream, сравнить tracked `cost_cents` (provider из `ImageResult`) с реальным OpenRouter биллингом. Настроить spend alerts.
**Why:** Новая внешняя API зависимость с per-image cost. При вирусном росте или abuse spend может резко вырасти.
**Effort:** S (ручной анализ + OpenRouter dashboard)
**Depends on:** Reference Images feature + 1 неделя данных

### Relationship Levels (Gamification)
**What:** Использовать `relationship_depth` (0-10) из `memory_models.py`. По мере общения уровень растёт → бот становится теплее, отправляет больше фото, пишет чаще первым.
**Why:** Retention mechanism — пользователь не хочет терять прогресс. Создаёт ощущение развивающихся отношений.
**Context:** Поле `relationship_depth` уже существует в `MemoryFact`. Нужно: логика увеличения уровня, влияние уровня на system prompt и proactive messaging частоту.
**Effort:** M (CC ~30 min)
**Depends on:** работающая память (memory writing) и проактивные сообщения

## P3 — Nice to Have

### Graph Memory для mem0 (Apache AGE / Kuzu)
**What:** Добавить graph memory backend к mem0 для multi-hop рассуждений о связях между фактами.
**Why:** Vector-only поиск не находит косвенные связи ("Алиса работает с Бобом → Боб любит кофе → принести кофе Алисе для Боба"). Graph memory добавляет entity-relationship traversal.
**Context:** mem0 поддерживает 6 graph backends: Neo4j, Memgraph, Kuzu (embedded), Apache AGE (Postgres extension), Neptune. Kuzu = zero infra. Но: +2x tokens per add(), баг с non-OpenAI providers для structuredLlm.
**Effort:** M (CC ~30 min)
**Depends on:** mem0 миграция (завершена)

### Remove get_*()/set_*() Singletons from Services
**What:** Удалить `get_*()` / `set_*()` из всех сервисов (llm_service, mem0_service, context_builder, langfuse_service, image_service, db_client, episode_manager). Все потребители уже используют constructor injection через `AppContext`.
**Why:** Двойной DI (composition root + модульные синглтоны) создаёт риск split-brain — два разных экземпляра сервиса в одном процессе.
**Context:** `ProactiveScheduler` всё ещё использует `get_*()` внутри методов. Нужно сначала рефакторить его на constructor injection, затем можно удалить все `get_*()/set_*()`.
**Effort:** S (CC ~15 min)
**Depends on:** Composition Root рефакторинг (завершён) + рефакторинг ProactiveScheduler

### Sprite Expansion Pipeline
**What:** Упростить добавление новых emotion sprites. Сейчас: сгенерировать через скрипт, загрузить в Supabase, обновить `SPRITE_EMOTIONS`, редеплой.
**Why:** Если пользователи просят эмоции не из набора (angry, sleepy, excited), turnaround медленный (code change + deploy).
**Effort:** S (CC ~15 min для admin скрипта)
**Depends on:** Reference Images feature

### Scaling Path Documentation
**What:** Документ с конкретными триггерами масштабирования: 100+ DAU → webhooks, 500+ DAU → отдельный proactive worker, 1000+ DAU → mem0 worker.
**Why:** Текущая архитектура (polling, single process) работает на ~100 пользователей. Нужно знать заранее, когда и что менять.
**Context:** Polling mode в aiogram, proactive scheduler в том же процессе, mem0 in-process. Каждый из этих компонентов — потенциальное bottleneck.
**Effort:** S (CC ~15 min, это документ, не код)
**Depends on:** ничего
