"""Tests for SillyTavern Character Card V2 parser."""

import base64
import json
import struct

import pytest

from bot.tools.import_card import (
    _parse_mes_example,
    extract_png_chara_data,
    parse_st_card,
    parse_st_card_json,
)


class TestParseStCardJson:
    """Tests for V2 JSON parsing."""

    def test_full_v2_card(self) -> None:
        data = {
            "data": {
                "name": "Alice",
                "description": "A friendly girl",
                "personality": "warm and kind",
                "scenario": "In a coffee shop",
                "first_mes": "Hello there!",
                "post_history_instructions": "Be casual",
                "mes_example": "<START>\nHi!\n<START>\nHow are you?",
            }
        }
        result = parse_st_card_json(data)
        assert result["name"] == "Alice"
        assert "A friendly girl" in result["personality"]
        assert "warm and kind" in result["personality"]
        assert "Сценарий: In a coffee shop" in result["personality"]
        assert result["greeting"] == "Hello there!"
        assert result["voice_style"] == "Be casual"
        assert len(result["example_messages"]) == 2

    def test_minimal_card(self) -> None:
        data = {"data": {"name": "Bob"}}
        result = parse_st_card_json(data)
        assert result["name"] == "Bob"
        assert result["greeting"] == ""
        assert result["example_messages"] == []

    def test_flat_v1_format(self) -> None:
        """V1 cards without 'data' wrapper are handled."""
        data = {"name": "Charlie", "description": "A tester", "first_mes": "Hey!"}
        result = parse_st_card_json(data)
        assert result["name"] == "Charlie"
        assert result["greeting"] == "Hey!"


class TestParseMesExample:
    """Tests for <START> block parsing."""

    def test_multiple_start_blocks(self) -> None:
        raw = "<START>\nHello!\nHow are you?\n<START>\nI'm fine."
        result = _parse_mes_example(raw)
        assert len(result) == 2
        assert "Hello!" in result[0]
        assert "I'm fine." in result[1]

    def test_empty_string(self) -> None:
        assert _parse_mes_example("") == []

    def test_none_like_empty(self) -> None:
        assert _parse_mes_example("   ") == []

    def test_single_block(self) -> None:
        raw = "<START>\nJust one example."
        result = _parse_mes_example(raw)
        assert len(result) == 1
        assert result[0] == "Just one example."


def _build_png_with_text_chunk(keyword: str, text: bytes) -> bytes:
    """Build a minimal PNG file with a tEXt chunk."""
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk (minimal valid)
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = _make_png_chunk(b"IHDR", ihdr_data)

    # tEXt chunk
    text_data = keyword.encode("latin-1") + b"\x00" + text
    text_chunk = _make_png_chunk(b"tEXt", text_data)

    # IEND chunk
    iend = _make_png_chunk(b"IEND", b"")

    return signature + ihdr + text_chunk + iend


def _make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a PNG chunk with length, type, data, and CRC."""
    import zlib

    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


class TestExtractPngCharaData:
    """Tests for PNG tEXt chunk extraction."""

    def test_extract_chara_from_text_chunk(self, tmp_path: pytest.TempPathFactory) -> None:
        card_data = {"data": {"name": "PngAlice", "first_mes": "Hi from PNG!"}}
        json_bytes = json.dumps(card_data).encode("utf-8")
        b64_text = base64.b64encode(json_bytes)

        png_bytes = _build_png_with_text_chunk("chara", b64_text)
        png_file = tmp_path / "test_card.png"  # type: ignore[operator]
        png_file.write_bytes(png_bytes)

        result = extract_png_chara_data(str(png_file))
        assert result["data"]["name"] == "PngAlice"

    def test_no_chara_chunk_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        png_bytes = _build_png_with_text_chunk("other", b"not a card")
        png_file = tmp_path / "no_chara.png"  # type: ignore[operator]
        png_file.write_bytes(png_bytes)

        with pytest.raises(ValueError, match="No 'chara' text chunk"):
            extract_png_chara_data(str(png_file))

    def test_not_a_png_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        bad_file = tmp_path / "not_png.png"  # type: ignore[operator]
        bad_file.write_bytes(b"this is not a png")

        with pytest.raises(ValueError, match="Not a valid PNG"):
            extract_png_chara_data(str(bad_file))


class TestParseStCard:
    """Tests for the dispatch function."""

    def test_dispatches_json(self, tmp_path: pytest.TempPathFactory) -> None:
        card = {"data": {"name": "JsonChar", "first_mes": "Hello!"}}
        json_file = tmp_path / "card.json"  # type: ignore[operator]
        json_file.write_text(json.dumps(card), encoding="utf-8")

        result = parse_st_card(str(json_file))
        assert result["name"] == "JsonChar"

    def test_dispatches_png(self, tmp_path: pytest.TempPathFactory) -> None:
        card = {"data": {"name": "PngChar"}}
        json_bytes = json.dumps(card).encode("utf-8")
        b64_text = base64.b64encode(json_bytes)
        png_bytes = _build_png_with_text_chunk("chara", b64_text)

        png_file = tmp_path / "card.png"  # type: ignore[operator]
        png_file.write_bytes(png_bytes)

        result = parse_st_card(str(png_file))
        assert result["name"] == "PngChar"

    def test_unsupported_extension_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        bad_file = tmp_path / "card.txt"  # type: ignore[operator]
        bad_file.write_text("not a card")

        with pytest.raises(ValueError, match="Unsupported file format"):
            parse_st_card(str(bad_file))
