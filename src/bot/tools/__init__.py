"""CLI tools for character management."""

from bot.tools.time_tool import GET_TIME_TOOL, execute_get_time
from bot.tools.weather_tool import GET_WEATHER_TOOL, execute_get_weather

__all__ = [
    "GET_TIME_TOOL",
    "GET_WEATHER_TOOL",
    "execute_get_time",
    "execute_get_weather",
]
