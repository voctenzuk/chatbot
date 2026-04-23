"""Weather tool using wttr.in (free, no API key)."""

import httpx
from loguru import logger

GET_WEATHER_TOOL: dict[str, object] = {
    "name": "get_weather",
    "description": (
        "Get current weather for a city. Use when the user asks about weather or temperature."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name (e.g., 'Москва', 'London')",
            }
        },
        "required": ["city"],
    },
}


async def execute_get_weather(city: str) -> str:
    """Fetch current weather from wttr.in. Returns Russian description."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://wttr.in/{city}",
                params={"format": "j1", "lang": "ru"},
            )
            resp.raise_for_status()
            data = resp.json()
            current = data["current_condition"][0]
            temp = current["temp_C"]
            feels = current.get("FeelsLikeC", temp)
            desc = current.get("lang_ru", [{}])
            desc_text = desc[0].get("value", current.get("weatherDesc", [{}])[0].get("value", ""))
            return f"{city}: {temp}°C (ощущается {feels}°C), {desc_text}"
    except Exception as exc:
        logger.warning("Weather API failed for {}: {}", city, exc)
        return f"Не удалось получить погоду для {city}"
