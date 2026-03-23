"""Tests for FactExtractorService."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.memory.fact_extractor import _EXTRACTION_PROMPT, ExtractedFact, FactExtractorService
from bot.memory.models import MemoryCategory, MemoryType, UserProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(content: str) -> AsyncMock:
    """Return a mock LLMService whose generate() returns the given content."""
    llm = MagicMock()
    response = MagicMock()
    response.content = content
    llm.generate = AsyncMock(return_value=response)
    return llm


def _json_facts(facts: list[dict]) -> str:
    return json.dumps({"facts": facts})


def _fact_dict(
    content: str = "Факт",
    category: str = "semantic",
    memory_type: str = "fact",
    importance: float = 1.0,
    emotional_valence: float = 0.0,
    tags: list[str] | None = None,
) -> dict:
    return {
        "content": content,
        "category": category,
        "memory_type": memory_type,
        "importance": importance,
        "emotional_valence": emotional_valence,
        "tags": tags or [],
    }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestFactExtractorServiceInit:
    def test_stores_llm(self) -> None:
        llm = MagicMock()
        service = FactExtractorService(llm=llm)
        assert service._llm is llm


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


class TestExtract:
    @pytest.mark.asyncio
    async def test_returns_empty_list_for_greeting(self) -> None:
        llm = _make_llm(_json_facts([]))
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Привет", "Привет! Как дела?")
        assert facts == []

    @pytest.mark.asyncio
    async def test_parses_identity_fact(self) -> None:
        raw = _json_facts(
            [_fact_dict("Имя пользователя — Саша", "semantic", "identity", 1.8, 0.2, ["name"])]
        )
        llm = _make_llm(raw)
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Меня зовут Саша", "Привет, Саша!")
        assert len(facts) == 1
        assert facts[0].content == "Имя пользователя — Саша"
        assert facts[0].category == MemoryCategory.SEMANTIC
        assert facts[0].memory_type == MemoryType.IDENTITY
        assert facts[0].importance == pytest.approx(1.8)
        assert facts[0].emotional_valence == pytest.approx(0.2)
        assert facts[0].tags == ["name"]

    @pytest.mark.asyncio
    async def test_calls_llm_with_system_and_user_message(self) -> None:
        llm = _make_llm(_json_facts([]))
        service = FactExtractorService(llm=llm)
        await service.extract("Тест", "Ответ")
        call_args = llm.generate.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Тест" in messages[1]["content"]
        assert "Ответ" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_includes_existing_profile_in_prompt(self) -> None:
        llm = _make_llm(_json_facts([]))
        service = FactExtractorService(llm=llm)
        profile = UserProfile(user_id=1, name="Алексей", occupation="инженер")
        await service.extract("Тест", "Ответ", existing_profile=profile)
        user_content = llm.generate.call_args[0][0][1]["content"]
        assert "Алексей" in user_content or "инженер" in user_content

    @pytest.mark.asyncio
    async def test_returns_empty_on_llm_failure(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Тест", "Ответ")
        assert facts == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self) -> None:
        llm = _make_llm("not valid json at all")
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Тест", "Ответ")
        assert facts == []

    @pytest.mark.asyncio
    async def test_strips_markdown_code_fences(self) -> None:
        inner = _json_facts([_fact_dict("Факт")])
        content = f"```json\n{inner}\n```"
        llm = _make_llm(content)
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Тест", "Ответ")
        assert len(facts) == 1
        assert facts[0].content == "Факт"

    @pytest.mark.asyncio
    async def test_multiple_facts_parsed(self) -> None:
        raw = _json_facts(
            [
                _fact_dict("Любит кофе", "preference", "like", 1.0, 0.3, ["coffee"]),
                _fact_dict("Не любит шум", "preference", "dislike", 1.0, -0.4),
            ]
        )
        llm = _make_llm(raw)
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Я люблю кофе, но не переношу шум", "Понятно!")
        assert len(facts) == 2
        assert facts[0].memory_type == MemoryType.LIKE
        assert facts[1].memory_type == MemoryType.DISLIKE

    @pytest.mark.asyncio
    async def test_importance_clamped_to_0_2(self) -> None:
        raw = _json_facts([_fact_dict(importance=99.9)])
        llm = _make_llm(raw)
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Тест", "Ответ")
        assert facts[0].importance == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_emotional_valence_clamped_to_minus1_1(self) -> None:
        raw = _json_facts([_fact_dict(emotional_valence=-5.0)])
        llm = _make_llm(raw)
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Тест", "Ответ")
        assert facts[0].emotional_valence == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_unknown_category_falls_back_to_semantic(self) -> None:
        raw = _json_facts([_fact_dict(category="unknown_cat")])
        llm = _make_llm(raw)
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Тест", "Ответ")
        assert facts[0].category == MemoryCategory.SEMANTIC

    @pytest.mark.asyncio
    async def test_unknown_memory_type_falls_back_to_fact(self) -> None:
        raw = _json_facts([_fact_dict(memory_type="unknown_type")])
        llm = _make_llm(raw)
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Тест", "Ответ")
        assert facts[0].memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_fact_without_content_skipped(self) -> None:
        raw = _json_facts([_fact_dict(content="")])
        llm = _make_llm(raw)
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Тест", "Ответ")
        assert facts == []

    @pytest.mark.asyncio
    async def test_non_dict_items_in_facts_list_skipped(self) -> None:
        raw = json.dumps({"facts": ["string item", None, _fact_dict("Валидный факт")]})
        llm = _make_llm(raw)
        service = FactExtractorService(llm=llm)
        facts = await service.extract("Тест", "Ответ")
        assert len(facts) == 1
        assert facts[0].content == "Валидный факт"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_extraction_prompt_is_string(self) -> None:
        assert isinstance(_EXTRACTION_PROMPT, str)
        assert len(_EXTRACTION_PROMPT) > 100

    def test_extraction_prompt_contains_categories(self) -> None:
        assert "semantic" in _EXTRACTION_PROMPT
        assert "emotional" in _EXTRACTION_PROMPT
        assert "preference" in _EXTRACTION_PROMPT

    def test_extraction_prompt_contains_json_format(self) -> None:
        assert '"facts"' in _EXTRACTION_PROMPT
        assert "emotional_valence" in _EXTRACTION_PROMPT

    def test_extraction_prompt_contains_russian_instructions(self) -> None:
        assert "Правила" in _EXTRACTION_PROMPT


# ---------------------------------------------------------------------------
# ExtractedFact dataclass
# ---------------------------------------------------------------------------


class TestExtractedFact:
    def test_defaults(self) -> None:
        fact = ExtractedFact(
            content="test",
            category=MemoryCategory.SEMANTIC,
            memory_type=MemoryType.FACT,
        )
        assert fact.importance == 1.0
        assert fact.emotional_valence == 0.0
        assert fact.tags == []

    def test_full_construction(self) -> None:
        fact = ExtractedFact(
            content="Любит Python",
            category=MemoryCategory.PREFERENCE,
            memory_type=MemoryType.LIKE,
            importance=1.5,
            emotional_valence=0.7,
            tags=["python", "programming"],
        )
        assert fact.content == "Любит Python"
        assert fact.category == MemoryCategory.PREFERENCE
        assert fact.memory_type == MemoryType.LIKE
        assert fact.importance == pytest.approx(1.5)
        assert fact.emotional_valence == pytest.approx(0.7)
        assert fact.tags == ["python", "programming"]
