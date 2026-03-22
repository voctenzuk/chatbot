"""Tests for CharacterConfig dataclass and DEFAULT_CHARACTER."""

import re

import pytest

from bot.character import DEFAULT_CHARACTER, SPRITE_EMOTIONS, CharacterConfig


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

    def test_default_character_reference_fields_present(self) -> None:
        """DEFAULT_CHARACTER has reference_image_url and sprite_urls attributes."""
        assert hasattr(DEFAULT_CHARACTER, "reference_image_url")
        assert hasattr(DEFAULT_CHARACTER, "sprite_urls")


class TestCharacterConfigNewFields:
    """Tests for reference image and sprite fields added in Visual Identity."""

    def test_accepts_reference_image_url(self) -> None:
        config = CharacterConfig(
            name="A",
            personality="P",
            appearance_en="E",
            voice_style="V",
            greeting="G",
            example_messages=[],
            reference_image_url="https://example.com/ref.png",
        )
        assert config.reference_image_url == "https://example.com/ref.png"

    def test_accepts_sprite_urls(self) -> None:
        sprites = {"smile": "https://storage.example.com/smile.png"}
        config = CharacterConfig(
            name="A",
            personality="P",
            appearance_en="E",
            voice_style="V",
            greeting="G",
            example_messages=[],
            sprite_urls=sprites,
        )
        assert config.sprite_urls == sprites

    def test_new_fields_default_to_none(self) -> None:
        config = CharacterConfig(
            name="A",
            personality="P",
            appearance_en="E",
            voice_style="V",
            greeting="G",
            example_messages=[],
        )
        assert config.reference_image_url is None
        assert config.sprite_urls is None


class TestSpriteEmotions:
    """Tests for SPRITE_EMOTIONS constant."""

    def test_is_tuple_of_strings(self) -> None:
        assert isinstance(SPRITE_EMOTIONS, tuple)
        assert len(SPRITE_EMOTIONS) == 5
        for emotion in SPRITE_EMOTIONS:
            assert isinstance(emotion, str)

    def test_contains_expected_emotions(self) -> None:
        assert set(SPRITE_EMOTIONS) == {"smile", "sad", "laugh", "thinking", "wink"}
