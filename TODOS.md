# TODOS

## P1 — Critical

### Unit Economics Validation
**What:** Проверить реальную цену kimi-k2p5 за токен против констант COST_PER_1M_INPUT/OUTPUT в `chat_pipeline.py`. После 1 недели данных — проанализировать cost_cents и скорректировать тиеры/лимиты.
**Why:** При оценочных ценах даже Pro тиер ($9.99) может быть убыточным. Нужны реальные данные для правильного прайсинга.
**Context:** `cost_cents NUMERIC(10,4)` уже трекается в `usage_tracking` (миграция 008). Константы: `COST_PER_1M_INPUT=0.15`, `COST_PER_1M_OUTPUT=0.60`, `IMAGE_COST_CENTS=5.0`. После накопления данных — сравнить с реальными API счетами.
**Effort:** S (ручной анализ данных после 1 недели)
**Depends on:** Character MVP (завершён)

## P2 — Important

### Image Consistency Evaluation
**What:** Оценить consistency внешности gpt-image-1 с appearance prefix после 50+ сгенерированных фото. Если <70% визуальной consistency — исследовать reference image подходы (Flux с IP-Adapter и т.д.).
**Why:** Appearance prefix в промпте даёт ~70-80% consistency по оценкам, но это не проверено на реальных данных. Консистентная внешность — core differentiator продукта.
**Context:** `ImageService.generate()` prepend'ит `CharacterConfig.appearance_en` к каждому промпту. Текущая строка: "Young woman, 24 years old, shoulder-length dark brown hair...". Альтернативы: Flux с reference image, DALL-E с style reference.
**Effort:** S (ручная оценка после 50+ фото)
**Depends on:** Character MVP (завершён)

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

### Scaling Path Documentation
**What:** Документ с конкретными триггерами масштабирования: 100+ DAU → webhooks, 500+ DAU → отдельный proactive worker, 1000+ DAU → mem0 worker.
**Why:** Текущая архитектура (polling, single process) работает на ~100 пользователей. Нужно знать заранее, когда и что менять.
**Context:** Polling mode в aiogram, proactive scheduler в том же процессе, mem0 in-process. Каждый из этих компонентов — потенциальное bottleneck.
**Effort:** S (CC ~15 min, это документ, не код)
**Depends on:** ничего
