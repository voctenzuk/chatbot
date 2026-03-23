"""Fact extractor service — LLM-powered structured fact extraction from conversations."""

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from bot.memory.models import MemoryCategory, MemoryType, UserProfile

if TYPE_CHECKING:
    from bot.llm.service import LLMService
    from bot.memory.mem0_service import Mem0MemoryService

_EXTRACTION_PROMPT = (
    "Ты — система извлечения фактов о пользователе из разговора.\n\n"
    "Проанализируй сообщение пользователя и ответ бота. "
    "Извлеки ТОЛЬКО новые факты о пользователе, которых нет в профиле.\n\n"
    "Категории (category):\n"
    "- semantic — общие факты (имя, возраст, профессия, место жительства)\n"
    "- emotional — эмоциональные состояния, реакции, чувства\n"
    "- preference — предпочтения (любит, не любит, интересы, хобби)\n"
    "- relationship — факты об отношениях с ботом (вехи, шутки, границы)\n"
    "- episodic — конкретные события, воспоминания\n"
    "- procedural — привычки, расписание, рутина\n\n"
    "Типы памяти (memory_type): fact, identity, goal, mood_state, like, dislike,\n"
    "topic_interest, milestone, habit, routine, emotional_reaction, stress_event,\n"
    "joy_event, communication_style, boundary, inside_joke\n\n"
    "Правила:\n"
    "1. Извлекай ТОЛЬКО факты о пользователе — не о боте\n"
    "2. Не дублируй факты из уже известного профиля\n"
    "3. importance: от 0.0 до 2.0 (identity/goal/milestone = 1.5+)\n"
    "4. emotional_valence: от -1.0 до +1.0, ОБЯЗАТЕЛЬНО заполни\n"
    "5. Если фактов нет — верни пустой список\n"
    "6. Отвечай ТОЛЬКО валидным JSON\n\n"
    'Формат: {"facts": [{"content": "...", "category": "semantic",\n'
    '"memory_type": "identity", "importance": 1.5, "emotional_valence": 0.1,\n'
    '"tags": ["name"]}]}\n\n'
    "Пример:\n"
    'Пользователь: "Меня зовут Саша, я программист"\n'
    'Бот: "Привет, Саша!"\n'
    "Уже известно: {}\n"
    '{"facts": [{"content": "Имя — Саша", "category": "semantic",\n'
    '"memory_type": "identity", "importance": 1.8, "emotional_valence": 0.2,\n'
    '"tags": ["name"]},\n'
    '{"content": "Работает программистом", "category": "semantic",\n'
    '"memory_type": "identity", "importance": 1.6, "emotional_valence": 0.1,\n'
    '"tags": ["job"]}]}\n\n'
    "Пример 2:\n"
    'Пользователь: "Привет"\n'
    'Бот: "Привет! Как дела?"\n'
    "Уже известно: {}\n"
    '{"facts": []}'
)


RELATED_MEMORY_THRESHOLD = 0.7


@dataclass
class ExtractedFact:
    """A single fact extracted from a conversation turn."""

    content: str
    category: MemoryCategory
    memory_type: MemoryType
    importance: float = 1.0
    emotional_valence: float = 0.0
    tags: list[str] = field(default_factory=list)
    related_memories: list[str] = field(default_factory=list)  # fact_ids from cross-search


class FactExtractorService:
    """LLM-powered fact extraction from conversation turns.

    Calls the LLM with a Russian-language structured extraction prompt and
    parses the JSON response into a list of ExtractedFact objects.
    Returns an empty list on any failure — safe for fire-and-forget usage.

    Optionally accepts a mem0_service for cross-memory reference population:
    after extracting facts, each fact is searched against existing memories;
    related fact IDs (similarity > RELATED_MEMORY_THRESHOLD) are stored in
    ExtractedFact.related_memories.
    """

    def __init__(self, llm: "LLMService", mem0_service: "Mem0MemoryService | None" = None) -> None:
        self._llm = llm
        self._mem0_service = mem0_service

    async def extract(
        self,
        user_message: str,
        bot_response: str,
        existing_profile: UserProfile | None = None,
        user_id: int = 0,
    ) -> list[ExtractedFact]:
        """Extract new facts from a single conversation turn.

        Args:
            user_message: The user's message text.
            bot_response: The bot's response text.
            existing_profile: Optional current profile to avoid duplicates.
            user_id: Telegram user ID for cross-memory search (0 if unknown).

        Returns:
            List of ExtractedFact objects, empty on any failure.
        """
        try:
            profile_summary = existing_profile.to_context_string() if existing_profile else ""
            user_content = (
                f"Пользователь: {user_message}\n"
                f"Бот: {bot_response}\n"
                f"Уже известно: {profile_summary or '{}'}"
            )

            messages = [
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user", "content": user_content},
            ]

            response = await self._llm.generate(messages)
            facts = self._parse_response(response.content)

        except Exception as exc:
            logger.warning("Fact extraction failed: {}", exc)
            return []

        # Cross-memory reference population: link new facts to related existing memories.
        if self._mem0_service and facts and user_id:
            for fact in facts:
                try:
                    related = await self._mem0_service.search(
                        query=fact.content, user_id=user_id, limit=3
                    )
                    for mem in related:
                        score = 0.0
                        if hasattr(mem, "metadata") and mem.metadata:
                            score = float(mem.metadata.get("score", 0.0))
                        # Also check importance_score as a proxy if score absent
                        if score == 0.0 and hasattr(mem, "importance_score"):
                            score = float(mem.importance_score)
                        if score > RELATED_MEMORY_THRESHOLD and hasattr(mem, "fact_id"):
                            fact_id: str = mem.fact_id
                            if fact_id and fact_id not in fact.related_memories:
                                fact.related_memories.append(fact_id)
                except Exception:
                    logger.debug("Cross-memory search failed for fact: {}", fact.content)

        return facts

    def _parse_response(self, raw: str) -> list[ExtractedFact]:
        """Parse LLM JSON response into ExtractedFact list."""
        try:
            text = raw.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()

            data = json.loads(text)
            raw_facts = data.get("facts", [])
            if not isinstance(raw_facts, list):
                return []

            result: list[ExtractedFact] = []
            for item in raw_facts:
                if not isinstance(item, dict):
                    continue
                fact = self._parse_fact(item)
                if fact is not None:
                    result.append(fact)
            return result

        except Exception as exc:
            logger.warning("Failed to parse fact extraction response: {}", exc)
            return []

    @staticmethod
    def _parse_fact(item: dict[str, Any]) -> "ExtractedFact | None":
        """Parse a single fact dict into an ExtractedFact."""
        content = item.get("content", "").strip()
        if not content:
            return None

        try:
            category = MemoryCategory(item.get("category", "semantic"))
        except ValueError:
            category = MemoryCategory.SEMANTIC

        try:
            memory_type = MemoryType(item.get("memory_type", "fact"))
        except ValueError:
            memory_type = MemoryType.FACT

        importance = float(item.get("importance", 1.0))
        importance = max(0.0, min(2.0, importance))

        emotional_valence = float(item.get("emotional_valence", 0.0))
        emotional_valence = max(-1.0, min(1.0, emotional_valence))

        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        return ExtractedFact(
            content=content,
            category=category,
            memory_type=memory_type,
            importance=importance,
            emotional_valence=emotional_valence,
            tags=[str(t) for t in tags],
        )
