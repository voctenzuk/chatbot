"""Framework-agnostic chat pipeline.

Extracts the core chat flow from handlers.py into a testable class
with zero aiogram dependencies. Receives all services via constructor.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from loguru import logger

from bot.conversation.context_builder import (
    ContextBuilder,
    ConversationMessage,
    MessageRole,
)
from bot.conversation.episode_manager import EpisodeManager, EpisodeMessage
from bot.conversation.system_prompt import get_system_prompt
from bot.llm.service import LLMResponse, LLMService, ToolCall
from bot.media.image_service import SEND_PHOTO_TOOL, SEND_SPRITE_TOOL, ImageResult
from bot.tools.time_tool import GET_TIME_TOOL, execute_get_time
from bot.tools.weather_tool import GET_WEATHER_TOOL, execute_get_weather

LLM_FALLBACK = "Прости, у меня сейчас не получается ответить. Попробуй ещё раз чуть позже."

# LLM cost per 1M tokens in cents (kimi-k2p5 pricing)
COST_PER_1M_INPUT = 0.15
COST_PER_1M_OUTPUT = 0.60

MAX_TOOL_ROUNDS = 3

_ROLE_MAP = {
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "system": MessageRole.SYSTEM,
}


@dataclass
class ChatResult:
    """Result of processing a chat message. Framework-agnostic."""

    response_text: str
    image_bytes: list[bytes] = field(default_factory=list)
    llm_response: LLMResponse | None = None
    was_rate_limited: bool = False


class ChatPipeline:
    """Core chat logic: receive message -> produce response.

    Zero aiogram imports. All services received via constructor.
    """

    def __init__(
        self,
        *,
        llm: LLMService,
        episode_manager: EpisodeManager,
        context_builder: ContextBuilder,
        langfuse: Any,
        memory: Any | None = None,
        image_service: Any | None = None,
        db_client: Any | None = None,
        character: Any | None = None,
    ) -> None:
        self._llm = llm
        self.episode_manager = episode_manager
        self._context_builder = context_builder
        self._langfuse = langfuse
        self._memory = memory
        self._image_service = image_service
        self.db_client = db_client
        self._character = character
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Always-available tools
        self._tool_registry: dict[str, Callable[..., Any]] = {
            "get_current_time": self._handle_get_time,
            "get_weather": self._handle_get_weather,
        }
        self._tool_schemas: list[dict[str, object]] = [GET_TIME_TOOL, GET_WEATHER_TOOL]

        # Image tools (only when image_service available)
        if self._image_service is not None:
            self._tool_registry["send_photo"] = self._handle_send_photo
            self._tool_registry["send_sprite"] = self._handle_send_sprite
            self._tool_schemas.extend([SEND_PHOTO_TOOL, SEND_SPRITE_TOOL])

    @property
    def character(self) -> Any:
        """Character configuration (CharacterConfig or None)."""
        return self._character

    @property
    def image_service(self) -> Any:
        """Image generation service (ImageService or None)."""
        return self._image_service

    def _fire_and_forget(self, coro: Any) -> None:
        """Schedule a coroutine as a background task with reference tracking."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)

    def _on_background_task_done(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Background task failed: {}", exc)

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
        result = await self.episode_manager.process_user_message(user_id=user_id, content=content)
        logger.debug(
            "User {} message persisted to episode {} (new_episode: {}, reason: {})",
            user_id,
            result.episode.id,
            result.is_new_episode,
            result.switch_decision.reason,
        )

        # 2. Check rate limit (fail open on errors)
        if self.db_client is not None:
            try:
                allowed = await self.db_client.check_rate_limit(user_id)
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
        generated_images: dict[str, Any] = {}
        try:
            llm_response = await self._generate_llm_response(
                user_id=user_id,
                content=content,
                user_name=user_name,
                generated_images=generated_images,
            )
            response = llm_response.content
        except Exception as llm_exc:
            logger.error("LLM generation failed for user {}: {}", user_id, llm_exc)
            response = LLM_FALLBACK
            llm_response = None

        # 4. Retrieve all cached images from tool loop
        all_images: list[bytes] = []
        image_cost_cents: float = 0.0
        if llm_response is not None and llm_response.tool_calls:
            for tool_call in llm_response.tool_calls:
                if tool_call.id in generated_images:
                    cached = generated_images.pop(tool_call.id)
                    if isinstance(cached, ImageResult):
                        all_images.append(cached.image_bytes)
                        image_cost_cents += cached.cost_cents
                    else:
                        # Raw bytes (e.g. sprites)
                        all_images.append(cached)

        # 5. Persist assistant response + track usage
        if llm_response is not None:
            await self.episode_manager.process_assistant_message(
                user_id=user_id,
                content=response,
                tokens_in=llm_response.tokens_in,
                tokens_out=llm_response.tokens_out,
                model=llm_response.model,
            )
        else:
            await self.episode_manager.process_assistant_message(
                user_id=user_id,
                content=response,
            )

        if self.db_client is not None and llm_response is not None:
            try:
                cost_cents = (
                    llm_response.tokens_in * COST_PER_1M_INPUT
                    + llm_response.tokens_out * COST_PER_1M_OUTPUT
                ) / 1_000_000
                cost_cents += image_cost_cents
                await self.db_client.increment_usage(
                    user_id,
                    msg_count=1,
                    tokens_in=llm_response.tokens_in,
                    tokens_out=llm_response.tokens_out,
                    cost_cents=cost_cents,
                )
            except Exception as exc:
                logger.warning("Usage tracking failed for user {}: {}", user_id, exc)

        # 6. Write to long-term memory (fire-and-forget)
        if self._memory is not None and llm_response is not None:
            try:
                memory_content = f"Пользователь: {content}\nПодруга: {response}"
                self._fire_and_forget(self._write_memory_background(memory_content, user_id))
            except Exception as exc:
                logger.warning("Memory write failed for user {}: {}", user_id, exc)

        return ChatResult(
            response_text=response,
            image_bytes=all_images,
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
        generated_images: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Build context and call LLM to produce a reply."""
        if generated_images is None:
            generated_images = {}

        episode = await self.episode_manager.get_current_episode(user_id)

        with self._langfuse.trace(
            user_id=user_id,
            session_id=episode.id if episode is not None else None,
            trace_name="chat",
        ):
            # Fetch semantic memories (best-effort)
            memories: list[Any] = []
            if self._memory is not None:
                try:
                    memories = await self._memory.search(
                        query=content,
                        user_id=user_id,
                        limit=5,
                    )
                except Exception as exc:
                    logger.warning("Memory search failed for user {}: {}", user_id, exc)

            # Fetch recent episode messages and convert
            recent_episode_msgs = await self.episode_manager.get_recent_messages(user_id, limit=20)
            recent_messages = _episode_messages_to_conversation(recent_episode_msgs)

            # Assemble LLM messages via context builder
            system_prompt = get_system_prompt(user_name=user_name, character=self._character)
            llm_messages = self._context_builder.assemble_for_llm(
                recent_messages=recent_messages,
                semantic_memories=memories,
                query=content,
                system_prompt=system_prompt,
            )

            # First LLM call (with all registered tools)
            llm_response = await self._llm.generate(llm_messages, tools=self._tool_schemas)

            # Multi-round tool execution loop
            all_tool_calls: list[ToolCall] = []
            total_tokens_in = llm_response.tokens_in
            total_tokens_out = llm_response.tokens_out
            current_messages = llm_messages

            for _round in range(MAX_TOOL_ROUNDS):
                if not llm_response.tool_calls:
                    break

                all_tool_calls.extend(llm_response.tool_calls)

                tool_messages: list[dict[str, str]] = []
                for tc in llm_response.tool_calls:
                    result_text = await self._execute_tool_for_loop(tc, user_id, generated_images)
                    tool_messages.append(
                        {
                            "role": "tool",
                            "content": result_text,
                            "tool_call_id": tc.id,
                        }
                    )

                current_messages = current_messages + [
                    {
                        "role": "assistant",
                        "content": llm_response.content or "",
                        "tool_calls": [
                            {"name": tc.name, "args": tc.args, "id": tc.id}
                            for tc in llm_response.tool_calls
                        ],
                    },
                    *tool_messages,
                ]

                llm_response = await self._llm.generate(current_messages)
                total_tokens_in += llm_response.tokens_in
                total_tokens_out += llm_response.tokens_out

            if all_tool_calls:
                return LLMResponse(
                    content=llm_response.content,
                    model=llm_response.model,
                    tokens_in=total_tokens_in,
                    tokens_out=total_tokens_out,
                    tool_calls=all_tool_calls,
                )

            return llm_response

    async def _execute_tool_for_loop(
        self,
        tool_call: ToolCall,
        user_id: int,
        generated_images: dict[str, Any],
    ) -> str:
        """Execute a tool call and return a result string for the LLM.

        Generated images are cached in generated_images so handle_message
        can deliver them without re-generating (avoids double API cost).
        Values are ImageResult (for photos) or raw bytes (for sprites).
        """
        handler = self._tool_registry.get(tool_call.name)
        if handler is None:
            return f"Unknown tool: {tool_call.name}"
        return await handler(tool_call, user_id, generated_images)

    async def _handle_send_photo(
        self,
        tool_call: ToolCall,
        user_id: int,
        generated_images: dict[str, Any],
    ) -> str:
        """Handle send_photo tool call."""
        try:
            if self._image_service is None:
                return "Image service unavailable"

            # Photo rate limit: fail-closed (no DB = no photos)
            if self.db_client is not None:
                allowed = await self.db_client.try_consume_photo(user_id)
                if not allowed:
                    return "Лимит фото на сегодня исчерпан. Напиши /upgrade для увеличения лимита."
            else:
                return "Image service unavailable"

            prompt = tool_call.args.get("prompt", "")
            image_result = await self._image_service.generate(prompt, user_id)
            if image_result is not None:
                generated_images[tool_call.id] = image_result
                return f"Photo generated successfully for prompt: {prompt[:50]}"
            return "Image generation failed"
        except Exception as e:
            return f"Image generation failed: {e}"

    async def _handle_send_sprite(
        self,
        tool_call: ToolCall,
        user_id: int,
        generated_images: dict[str, Any],
    ) -> str:
        """Handle send_sprite tool call."""
        try:
            if self._image_service is None:
                return "Sprite service unavailable"
            emotion = tool_call.args.get("emotion", "smile")
            sprite_bytes = await self._image_service.get_sprite(emotion)
            if sprite_bytes is not None:
                generated_images[tool_call.id] = sprite_bytes  # raw bytes, no cost
                return f"Sent emotion sprite: {emotion}"
            return "Sprite unavailable"
        except Exception as e:
            return f"Sprite failed: {e}"

    async def _handle_get_time(
        self,
        tool_call: ToolCall,
        user_id: int,
        generated_images: dict[str, Any],
    ) -> str:
        """Handle get_current_time tool call."""
        return await execute_get_time()

    async def _handle_get_weather(
        self,
        tool_call: ToolCall,
        user_id: int,
        generated_images: dict[str, Any],
    ) -> str:
        """Handle get_weather tool call."""
        city = tool_call.args.get("city", "")
        return await execute_get_weather(city)

    async def _write_memory_background(self, memory_content: str, user_id: int) -> None:
        """Write to long-term memory in background. Fire-and-forget."""
        if self._memory is None:
            return
        try:
            await self._memory.write_factual(content=memory_content, user_id=user_id)
        except Exception as exc:
            logger.warning("Background memory write failed for user {}: {}", user_id, exc)


def _episode_messages_to_conversation(
    messages: list[EpisodeMessage],
) -> list[ConversationMessage]:
    """Convert EpisodeMessage list to ConversationMessage list."""
    return [
        ConversationMessage(
            role=_ROLE_MAP.get(msg.role, MessageRole.USER),
            content=msg.content_text,
            timestamp=msg.created_at or datetime.now(tz=UTC),
        )
        for msg in messages
    ]
