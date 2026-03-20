"""Framework-agnostic chat pipeline.

Extracts the core chat flow from handlers.py into a testable class
with zero aiogram dependencies. Receives all services via constructor.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from bot.conversation.context_builder import (
    ConversationMessage,
    MessageRole,
    get_context_builder,
)
from bot.conversation.episode_manager import EpisodeManager, EpisodeMessage
from bot.conversation.system_prompt import get_system_prompt
from bot.infra.langfuse_service import get_langfuse_service
from bot.llm.service import LLMResponse, ToolCall, get_llm_service

try:
    from bot.memory.cognee_service import get_memory_service

    MEMORY_SERVICE_AVAILABLE = True
except ImportError:
    get_memory_service = None  # type: ignore[assignment]
    MEMORY_SERVICE_AVAILABLE = False

try:
    from bot.infra.db_client import get_db_client

    DB_CLIENT_AVAILABLE = True
except ImportError:
    get_db_client = None  # type: ignore[assignment]
    DB_CLIENT_AVAILABLE = False

try:
    from bot.media.image_service import SEND_PHOTO_TOOL, get_image_service

    IMAGE_SERVICE_AVAILABLE = True
except ImportError:
    SEND_PHOTO_TOOL = None  # type: ignore[assignment]
    get_image_service = None  # type: ignore[assignment]
    IMAGE_SERVICE_AVAILABLE = False


_LLM_FALLBACK = "Прости, у меня сейчас не получается ответить. Попробуй ещё раз чуть позже."

_ROLE_MAP = {
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "system": MessageRole.SYSTEM,
}

# Counter for scheduling periodic cognify() calls
_memory_write_counts: dict[int, int] = {}
_COGNIFY_EVERY_N_WRITES: int = 10


@dataclass
class ChatResult:
    """Result of processing a chat message. Framework-agnostic."""

    response_text: str
    image_bytes: bytes | None = None
    llm_response: LLMResponse | None = None
    was_rate_limited: bool = False


class ChatPipeline:
    """Core chat logic: receive message -> produce response.

    Zero aiogram imports. All services received via constructor or
    module-level singletons (get_*() pattern).
    """

    def __init__(self, episode_manager: EpisodeManager) -> None:
        self._episode_manager = episode_manager
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._generated_images: dict[str, bytes] = {}  # cache from tool loop

    def _fire_and_forget(self, coro: Any) -> None:
        """Schedule a coroutine as a background task with reference tracking."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def handle_message(
        self,
        user_id: int,
        content: str,
        user_name: str | None = None,
    ) -> ChatResult:
        """Process a user message and return a response.

        This is the main pipeline:
        1. Persist user message (episode manager)
        2. Check rate limit (fail-open)
        3. Generate LLM response (with memory search + context assembly)
        4. Handle tool calls (image generation)
        5. Persist assistant response + track usage (post-send)
        6. Write to long-term memory (fire-and-forget)
        """
        # 1. Persist user message
        result = await self._episode_manager.process_user_message(user_id=user_id, content=content)
        logger.debug(
            "User {} message persisted to episode {} (new_episode: {}, reason: {})",
            user_id,
            result.episode.id,
            result.is_new_episode,
            result.switch_decision.reason,
        )

        # 2. Check rate limit (fail open on errors)
        if DB_CLIENT_AVAILABLE and get_db_client is not None:
            try:
                db = get_db_client()
                allowed = await db.check_rate_limit(user_id)
                if not allowed:
                    return ChatResult(
                        response_text=(
                            "Сегодня лимит сообщений исчерпан 😔\n"
                            "Напиши /upgrade чтобы увеличить лимит."
                        ),
                        was_rate_limited=True,
                    )
            except Exception as exc:
                logger.warning("Rate limit check failed for user {}, allowing: {}", user_id, exc)

        # 3. Generate LLM response
        try:
            llm_response = await self._generate_llm_response(
                user_id=user_id,
                content=content,
                user_name=user_name,
            )
            response = llm_response.content
        except Exception as llm_exc:
            logger.error("LLM generation failed for user {}: {}", user_id, llm_exc)
            response = _LLM_FALLBACK
            llm_response = None

        # 4. Retrieve cached image from tool loop (already generated in _execute_tool_for_loop)
        image_bytes: bytes | None = None
        if llm_response is not None and llm_response.tool_calls:
            for tool_call in llm_response.tool_calls:
                if tool_call.name == "send_photo" and tool_call.id in self._generated_images:
                    image_bytes = self._generated_images.pop(tool_call.id)
                    break

        # 5. Persist assistant response + track usage
        if llm_response is not None:
            await self._episode_manager.process_assistant_message(
                user_id=user_id,
                content=response,
                tokens_in=llm_response.tokens_in,
                tokens_out=llm_response.tokens_out,
                model=llm_response.model,
            )
        else:
            await self._episode_manager.process_assistant_message(
                user_id=user_id,
                content=response,
            )

        if DB_CLIENT_AVAILABLE and get_db_client is not None and llm_response is not None:
            try:
                db = get_db_client()
                await db.increment_usage(
                    user_id,
                    msg_count=1,
                    tokens_in=llm_response.tokens_in,
                    tokens_out=llm_response.tokens_out,
                )
            except Exception as exc:
                logger.warning("Usage tracking failed for user {}: {}", user_id, exc)

        # 6. Write to long-term memory (fire-and-forget)
        if MEMORY_SERVICE_AVAILABLE and get_memory_service is not None and llm_response is not None:
            try:
                mem_service = get_memory_service()
                memory_content = f"Пользователь: {content}\nПодруга: {response}"
                self._fire_and_forget(
                    self._write_memory_background(mem_service, memory_content, user_id)
                )
            except Exception as exc:
                logger.warning("Memory write failed for user {}: {}", user_id, exc)

        return ChatResult(
            response_text=response,
            image_bytes=image_bytes,
            llm_response=llm_response,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_llm_response(
        self,
        user_id: int,
        content: str,
        user_name: str | None,
    ) -> LLMResponse:
        """Build context and call LLM to produce a reply."""
        episode = await self._episode_manager.get_current_episode(user_id)
        lf_config = get_langfuse_service().create_config(
            user_id=user_id,
            session_id=episode.id if episode is not None else None,
            trace_name="chat",
        )

        # Fetch semantic memories (best-effort)
        memories = []
        if MEMORY_SERVICE_AVAILABLE and get_memory_service is not None:
            try:
                memories = await get_memory_service().search(
                    query=content,
                    user_id=user_id,
                    limit=5,
                )
            except Exception as exc:
                logger.warning("Memory search failed for user {}: {}", user_id, exc)

        # Fetch recent episode messages and convert
        recent_episode_msgs = await self._episode_manager.get_recent_messages(user_id, limit=20)
        recent_messages = _episode_messages_to_conversation(recent_episode_msgs)

        # Assemble LLM messages via context builder
        system_prompt = get_system_prompt(user_name=user_name)
        llm_messages = get_context_builder().assemble_for_llm(
            recent_messages=recent_messages,
            semantic_memories=memories,
            query=content,
            system_prompt=system_prompt,
        )

        # First LLM call (with image tool if available)
        tools = [SEND_PHOTO_TOOL] if IMAGE_SERVICE_AVAILABLE and SEND_PHOTO_TOOL else None
        llm_svc = get_llm_service()
        llm_response = await llm_svc.generate(llm_messages, tools=tools, config=lf_config)

        # Tool execution loop
        if llm_response.tool_calls and tools:
            tool_messages: list[dict[str, str]] = []
            for tc in llm_response.tool_calls:
                result_text = await self._execute_tool_for_loop(tc, user_id)
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": result_text,
                        "tool_call_id": tc.id,
                    }
                )

            follow_up: list[dict[str, str]] = llm_messages + [
                {
                    "role": "assistant",
                    "content": llm_response.content or "",
                    "tool_calls": [  # type: ignore[list-item]
                        {"name": tc.name, "args": tc.args, "id": tc.id}
                        for tc in llm_response.tool_calls
                    ],
                },
                *tool_messages,
            ]

            final_response = await llm_svc.generate(follow_up, config=lf_config)

            return LLMResponse(
                content=final_response.content,
                model=final_response.model,
                tokens_in=llm_response.tokens_in + final_response.tokens_in,
                tokens_out=llm_response.tokens_out + final_response.tokens_out,
                tool_calls=llm_response.tool_calls,
            )

        return llm_response

    async def _execute_tool_for_loop(self, tool_call: ToolCall, user_id: int) -> str:
        """Execute a tool call and return a result string for the LLM.

        Generated images are cached in self._generated_images so handle_message
        can deliver them without re-generating (avoids double API cost).
        """
        if tool_call.name == "send_photo":
            try:
                if IMAGE_SERVICE_AVAILABLE and get_image_service is not None:
                    prompt = tool_call.args.get("prompt", "")
                    image_bytes = await get_image_service().generate(prompt, user_id)
                    if image_bytes is not None:
                        self._generated_images[tool_call.id] = image_bytes
                        return f"Photo generated successfully for prompt: {prompt[:50]}"
                return "Image service unavailable"
            except Exception as e:
                return f"Image generation failed: {e}"
        return f"Unknown tool: {tool_call.name}"

    async def _write_memory_background(
        self, mem_service: Any, memory_content: str, user_id: int
    ) -> None:
        """Write to long-term memory in background. Fire-and-forget."""
        try:
            await mem_service.write_factual(content=memory_content, user_id=user_id)

            _memory_write_counts[user_id] = _memory_write_counts.get(user_id, 0) + 1
            if _memory_write_counts[user_id] >= _COGNIFY_EVERY_N_WRITES:
                _memory_write_counts[user_id] = 0
                self._fire_and_forget(self._run_cognify_background())
        except Exception as exc:
            logger.warning("Background memory write failed for user {}: {}", user_id, exc)

    async def _run_cognify_background(self) -> None:
        """Run cognify in background to build knowledge graph."""
        try:
            if MEMORY_SERVICE_AVAILABLE and get_memory_service is not None:
                await get_memory_service().cognify()
                logger.info("Background cognify completed successfully")
        except Exception as exc:
            logger.warning("Background cognify failed: {}", exc)


def _episode_messages_to_conversation(
    messages: list[EpisodeMessage],
) -> list[ConversationMessage]:
    """Convert EpisodeMessage list to ConversationMessage list."""
    return [
        ConversationMessage(
            role=_ROLE_MAP.get(msg.role, MessageRole.USER),
            content=msg.content_text,
            timestamp=msg.created_at or datetime.now(),
        )
        for msg in messages
    ]
