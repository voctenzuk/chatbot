"""Current time tool for LLM function calling."""

from datetime import datetime, timedelta, timezone

MSK = timezone(timedelta(hours=3))

GET_TIME_TOOL: dict[str, object] = {
    "name": "get_current_time",
    "description": (
        "Get the current date and time in Moscow. Use when the user asks what time or date it is."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


async def execute_get_time() -> str:
    """Return current Moscow time as a formatted string."""
    now = datetime.now(tz=MSK)
    return now.strftime("%H:%M, %d %B %Y (МСК)")
