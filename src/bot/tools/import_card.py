"""SillyTavern Character Card V2 parser.

Parses .json and .png character cards into CharacterConfig field values.
CLI usage: uv run python -m bot.tools.import_card path/to/card.json

PNG cards store character data in a tEXt chunk with keyword "chara",
base64-encoded JSON payload.
"""

import argparse
import base64
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any


def parse_st_card_json(data: dict[str, Any]) -> dict[str, Any]:
    """Map SillyTavern V2 card fields to CharacterConfig-compatible dict.

    Handles both V2 format (with "data" wrapper) and flat V1 format.
    """
    # V2 cards wrap fields under "data"
    card = data.get("data", data)

    name = card.get("name", "Unknown")

    # Personality: merge description + personality
    parts = []
    if card.get("description"):
        parts.append(card["description"])
    if card.get("personality"):
        parts.append(card["personality"])
    if card.get("scenario"):
        parts.append(f"\nСценарий: {card['scenario']}")
    personality = "\n\n".join(parts) if parts else ""

    greeting = card.get("first_mes", "")
    voice_style = card.get("post_history_instructions", "")
    example_messages = _parse_mes_example(card.get("mes_example", ""))

    return {
        "name": name,
        "personality": personality,
        "greeting": greeting,
        "voice_style": voice_style,
        "example_messages": example_messages,
        "appearance_en": "",  # Must be filled manually or via LLM extraction
    }


def _parse_mes_example(raw: str) -> list[str]:
    """Parse <START> delimited example messages into a list of strings."""
    if not raw or not raw.strip():
        return []

    blocks: list[str] = []
    current: list[str] = []

    for line in raw.strip().splitlines():
        stripped = line.strip()
        if stripped.upper() == "<START>":
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    return [b for b in blocks if b]


def extract_png_chara_data(png_path: str) -> dict[str, Any]:
    """Extract character data from PNG tEXt chunk with keyword 'chara'.

    Uses stdlib struct for PNG chunk reading (no Pillow dependency).
    """
    path = Path(png_path)
    data = path.read_bytes()

    # Verify PNG signature
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        msg = f"Not a valid PNG file: {png_path}"
        raise ValueError(msg)

    offset = 8  # Skip signature
    while offset < len(data):
        if offset + 8 > len(data):
            break

        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]

        if chunk_type == b"tEXt":
            # tEXt: keyword\x00text
            null_idx = chunk_data.find(b"\x00")
            if null_idx >= 0:
                keyword = chunk_data[:null_idx].decode("latin-1")
                if keyword == "chara":
                    text_data = chunk_data[null_idx + 1 :]
                    json_str = base64.b64decode(text_data).decode("utf-8")
                    return json.loads(json_str)

        elif chunk_type == b"iTXt":
            # iTXt: keyword\x00compression_flag\x00compression_method\x00lang\x00translated\x00text
            null_idx = chunk_data.find(b"\x00")
            if null_idx >= 0:
                keyword = chunk_data[:null_idx].decode("latin-1")
                if keyword == "chara":
                    rest = chunk_data[null_idx + 1 :]
                    compression_flag = rest[0]
                    # Skip compression_method(1), then find lang\0translated\0
                    rest = rest[2:]  # skip flag + method
                    null1 = rest.find(b"\x00")
                    rest = rest[null1 + 1 :] if null1 >= 0 else rest
                    null2 = rest.find(b"\x00")
                    text_data = rest[null2 + 1 :] if null2 >= 0 else rest

                    if compression_flag:
                        text_data = zlib.decompress(text_data)

                    json_str = base64.b64decode(text_data).decode("utf-8")
                    return json.loads(json_str)

        elif chunk_type == b"zTXt":
            # zTXt: keyword\x00compression_method\x00compressed_text
            null_idx = chunk_data.find(b"\x00")
            if null_idx >= 0:
                keyword = chunk_data[:null_idx].decode("latin-1")
                if keyword == "chara":
                    compressed = chunk_data[null_idx + 2 :]  # skip null + compression method
                    text_data = zlib.decompress(compressed)
                    json_str = base64.b64decode(text_data).decode("utf-8")
                    return json.loads(json_str)

        # Move to next chunk: length(4) + type(4) + data(length) + CRC(4)
        offset += 12 + length

    msg = f"No 'chara' text chunk found in PNG: {png_path}"
    raise ValueError(msg)


def parse_st_card(path: str) -> dict[str, Any]:
    """Parse a SillyTavern character card from .json or .png file."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        return parse_st_card_json(raw)
    elif p.suffix.lower() == ".png":
        raw = extract_png_chara_data(path)
        return parse_st_card_json(raw)
    else:
        msg = f"Unsupported file format: {p.suffix}. Use .json or .png."
        raise ValueError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse SillyTavern character card")
    parser.add_argument("path", help="Path to .json or .png character card")
    args = parser.parse_args()

    try:
        result = parse_st_card(args.path)
        print("=== Parsed Character Card ===")  # noqa: T201
        for key, value in result.items():
            if isinstance(value, list):
                print(f"{key}: [{len(value)} items]")  # noqa: T201
                for i, item in enumerate(value):
                    print(f"  [{i}] {item[:80]}...")  # noqa: T201
            elif isinstance(value, str) and len(value) > 100:
                print(f"{key}: {value[:100]}...")  # noqa: T201
            else:
                print(f"{key}: {value}")  # noqa: T201
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


if __name__ == "__main__":
    main()
