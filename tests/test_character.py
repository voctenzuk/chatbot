"""Tests for CharacterConfig dataclass and DEFAULT_CHARACTER."""

import re

import pytest

from bot.character import DEFAULT_CHARACTER, CharacterConfig


class TestDefaultCharacter:
    """Tests for the DEFAULT_CHARACTER singleton."""

    def test_default_character_has_all_required_fields(self) -> None:
        """DEFAULT_CHARACTER must have all fields populated with non-empty values."""
        assert DEFAULT_CHARACTER.name
        assert DEFAULT_CHARACTER.personality
        assert DEFAULT_CHARACTER.appearance_en
        assert DEFAULT_CHARACTER.voice_style
        assert DEFAULT_CHARACTER.greeting
        assert DEFAULT_CHARACTER.example_messages

    def test_character_config_is_frozen(self) -> None:
        """CharacterConfig is frozen — attribute assignment must raise."""
        with pytest.raises(AttributeError):
            DEFAULT_CHARACTER.name = "Другое имя"  # type: ignore[misc]

    def test_appearance_en_is_english(self) -> None:
        """appearance_en must contain only English text (no Cyrillic characters)."""
        cyrillic_re = re.compile(r"[\u0400-\u04FF]")
        assert not cyrillic_re.search(DEFAULT_CHARACTER.appearance_en), (
            f"appearance_en contains Cyrillic: {DEFAULT_CHARACTER.appearance_en}"
        )

    def test_example_messages_is_nonempty_list_of_strings(self) -> None:
        """example_messages must be a non-empty list where every element is a string."""
        assert isinstance(DEFAULT_CHARACTER.example_messages, list)
        assert len(DEFAULT_CHARACTER.example_messages) > 0
        for msg in DEFAULT_CHARACTER.example_messages:
            assert isinstance(msg, str)
            assert msg  # non-empty

    def test_character_config_creation_with_custom_values(self) -> None:
        """CharacterConfig can be created with fully custom values."""
        custom = CharacterConfig(
            name="Тест",
            personality="Тестовая личность",
            appearance_en="Blue eyes, blond hair",
            voice_style="Формальный стиль",
            greeting="Привет тестовый",
            example_messages=["Пример 1", "Пример 2"],
        )
        assert custom.name == "Тест"
        assert custom.personality == "Тестовая личность"
        assert custom.appearance_en == "Blue eyes, blond hair"
        assert custom.voice_style == "Формальный стиль"
        assert custom.greeting == "Привет тестовый"
        assert custom.example_messages == ["Пример 1", "Пример 2"]
