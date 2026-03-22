# Настройка персонажа

Бот использует `CharacterConfig` — frozen dataclass в `src/bot/character.py`. Сейчас один персонаж (`DEFAULT_CHARACTER`), позже — multi-character.

## Поля CharacterConfig

| Поле | Язык | Что делает |
|------|------|------------|
| `name` | Рус | Имя персонажа, видно пользователю |
| `personality` | Рус | Описание личности → идёт в system prompt |
| `appearance_en` | **Англ** | Описание внешности → идёт в prompt для image generation. Английский потому что image models (SeeDream, FLUX) работают значительно лучше с английскими промптами |
| `voice_style` | Рус | Стиль общения → идёт в system prompt |
| `greeting` | Рус | Первое сообщение при `/start` для новых пользователей |
| `example_messages` | Рус | Few-shot примеры ответов → идёт в system prompt (+100-150 токенов на пример) |
| `reference_image_url` | — | URL reference фото в Supabase Storage для identity-preserving генерации |
| `sprite_urls` | — | Маппинг `{эмоция: URL}` для пре-генерированных спрайтов |

## Способ 1: Редактирование напрямую

Открой `src/bot/character.py` и измени `DEFAULT_CHARACTER`:

```python
DEFAULT_CHARACTER = CharacterConfig(
    name="Алиса",
    personality=(
        "Ты — Алиса, виртуальная подруга в Telegram. "
        "Тебе 24 года, ты живёшь в Москве, работаешь дизайнером. "
        # ... описание личности
    ),
    appearance_en=(
        "Young woman, 24 years old, shoulder-length dark brown hair, "
        "warm brown eyes, slim build, European features"
    ),
    voice_style=(
        "Общайся как близкая подруга: тепло, искренне, иногда с лёгкой иронией."
    ),
    greeting="Привет! Я Алиса 🙂\nРада познакомиться!",
    example_messages=[
        "Ого, серьёзно? Расскажи подробнее 😊",
        "Слушай, я тебя понимаю. Хочешь поговорить об этом?",
    ],
)
```

## Способ 2: Импорт из SillyTavern карты

### Что такое ST Card

SillyTavern character card — PNG-файл с встроенными JSON-метаданными (tEXt chunk `chara`, base64). Стандарт де-факто для AI roleplay персонажей. Также бывают в формате `.json`.

### Где скачать карты

| Сайт | Карты | Особенности |
|------|-------|-------------|
| [CharaVault](https://charavault.net/) | 195K+ | Фильтры, NSFW toggle, прямые скачивания |
| [AI Character Cards](https://aicharactercards.com/) | Курируемая | Гайды по созданию, community |
| [CharacterCard.com](https://charactercard.com/download) | PNG формат | Без регистрации |

### Импорт

```bash
# Из PNG карты (данные в tEXt chunk)
uv run python -m bot.tools.import_card path/to/card.png

# Из JSON
uv run python -m bot.tools.import_card path/to/card.json
```

Парсер выведет поля `CharacterConfig`:
```
=== Parsed Character Card ===
name: Alice
personality: Alice is a cheerful 24-year-old...
greeting: Hey there! I'm Alice...
voice_style: Speak warmly and casually...
example_messages: [3 items]
  [0] Oh wow, tell me more!...
  [1] I totally get that...
appearance_en:
```

### Маппинг полей ST Card V2 → CharacterConfig

| ST Card V2 | → | CharacterConfig |
|------------|---|-----------------|
| `name` | → | `name` |
| `description` + `personality` + `scenario` | → | `personality` (объединяются) |
| `first_mes` | → | `greeting` |
| `post_history_instructions` | → | `voice_style` |
| `mes_example` (блоки `<START>`) | → | `example_messages` (парсятся в list) |
| Аватар из PNG | → | Можно загрузить как `reference_image_url` |
| — | — | `appearance_en` **нужно заполнить вручную** |

### После импорта

1. Скопируй вывод в `src/bot/character.py` → `DEFAULT_CHARACTER`
2. **Обязательно допиши `appearance_en`** — английское описание внешности. Без него фото будут генерироваться без reference к внешности персонажа
3. При необходимости переведи `personality` и `voice_style` на русский (ST карты обычно на английском)

## Визуальная идентичность (опционально)

### Reference Image

Для consistent фото персонажа можно загрузить reference image:

1. Сгенерируй или подбери фото персонажа (крупный план лица)
2. Загрузи в Supabase Storage → bucket `character-assets` → `{name}/reference.png`
3. Укажи URL в `reference_image_url`

При генерации фото через `send_photo` бот отправит reference image вместе с промптом для identity-preserving генерации (SeeDream 4.5).

### Эмоциональные спрайты

Пре-генерированные фото с разными эмоциями для мгновенных реакций ($0, 100% consistency):

Доступные эмоции (`SPRITE_EMOTIONS`): `smile`, `sad`, `laugh`, `thinking`, `wink`

1. Сгенерируй спрайт для каждой эмоции
2. Загрузи в Supabase Storage → `character-assets/{name}/sprites/{emotion}.png`
3. Укажи URLs в `sprite_urls`:

```python
DEFAULT_CHARACTER = CharacterConfig(
    # ...
    sprite_urls={
        "smile": "https://your-project.supabase.co/storage/v1/object/public/character-assets/alice/sprites/smile.png",
        "sad": "https://..../sad.png",
        "laugh": "https://..../laugh.png",
        "thinking": "https://..../thinking.png",
        "wink": "https://..../wink.png",
    },
)
```

LLM сам выбирает между `send_sprite` (мгновенно, $0) и `send_photo` (8-10с, ~$0.04) через tool calling.

## Поддерживаемые форматы

| Формат | Парсинг | Описание |
|--------|---------|----------|
| `.json` | Прямой JSON | V1 (flat) и V2 (с `data` wrapper) |
| `.png` tEXt | base64 → JSON | Стандартный ST Card формат |
| `.png` iTXt | base64 → JSON | С поддержкой сжатия (zlib) |
| `.png` zTXt | zlib + base64 → JSON | Сжатые карты |
