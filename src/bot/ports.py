"""Port protocols for the chatbot's swappable infrastructure boundaries.

Only services where implementation swapping is realistic get a Protocol:
- LLMPort: swap LLM provider (OpenAI, Anthropic, local Ollama)
- MemoryPort: swap memory backend (Cognee, in-memory, future alternatives)
- MessageDeliveryPort: swap delivery channel (Telegram, Web, CLI)
"""

from typing import Any, Protocol, runtime_checkable

from bot.llm.service import LLMResponse
from bot.memory.models import MemoryFact, MemoryType


@runtime_checkable
class LLMPort(Protocol):
    """Generate text responses from an LLM."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


@runtime_checkable
class MemoryPort(Protocol):
    """Read and write long-term semantic memory."""

    async def search(
        self,
        query: str,
        user_id: int,
        run_id: str | None = None,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryFact]: ...

    async def write_factual(
        self,
        content: str,
        user_id: int,
        metadata: dict[str, Any] | None = None,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 1.0,
        tags: list[str] | None = None,
    ) -> str: ...

    async def cognify(self) -> None: ...


@runtime_checkable
class MessageDeliveryPort(Protocol):
    """Send messages to a user (Telegram, Web, etc.)."""

    async def send_text(self, chat_id: int, text: str) -> None: ...

    async def send_photo(
        self, chat_id: int, photo_bytes: bytes, caption: str | None = None
    ) -> None: ...
