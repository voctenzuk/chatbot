"""Telegram bot handlers with episode-aware message persistence and LLM integration."""

from __future__ import annotations

from datetime import datetime

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from loguru import logger

from bot.services.context_builder import (
    ConversationMessage,
    MessageRole,
    get_context_builder,
)
from bot.services.episode_manager import (
    EpisodeManager,
    EpisodeMessage,
    get_episode_manager,
    set_episode_manager,
)
from bot.services.llm_service import LLMResponse, get_llm_service
from bot.services.system_prompt import get_system_prompt

try:
    from bot.services.cognee_memory_service import get_memory_service

    MEMORY_SERVICE_AVAILABLE = True
except ImportError:
    get_memory_service = None  # type: ignore[assignment]
    MEMORY_SERVICE_AVAILABLE = False

try:
    from bot.services.db_client import get_db_client

    DB_CLIENT_AVAILABLE = True
except ImportError:
    get_db_client = None  # type: ignore
    DB_CLIENT_AVAILABLE = False

router = Router()

_LLM_FALLBACK = "Прости, у меня сейчас не получается ответить. Попробуй ещё раз чуть позже."
_ROLE_MAP = {
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "system": MessageRole.SYSTEM,
}


class EpisodeManagerUnavailableError(Exception):
    """Raised when episode manager cannot be initialized."""

    def __init__(
        self, message: str = "Message persistence unavailable", cause: Exception | None = None
    ) -> None:
        super().__init__(message)
        self.cause = cause


async def get_episode_manager_service() -> EpisodeManager:
    """Get or initialize episode manager service."""
    manager = get_episode_manager()

    manager_db = getattr(manager, "db", None) if manager is not None else None
    if manager is not None and manager_db is None and DB_CLIENT_AVAILABLE:
        try:
            if get_db_client is not None:
                db_client = get_db_client()
                manager = EpisodeManager(db_client=db_client)
                set_episode_manager(manager)
        except Exception as e:
            logger.error(
                "Failed to initialize database client for episode manager: {}. "
                "Message persistence will be unavailable.",
                e,
            )
            raise EpisodeManagerUnavailableError(
                "Database initialization failed - messages cannot be persisted",
                cause=e,
            ) from e

    if manager is None:
        manager = EpisodeManager()
        set_episode_manager(manager)
    return manager


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


async def _generate_llm_response(
    user_id: int,
    content: str,
    user_name: str | None,
    episode_manager: EpisodeManager,
) -> LLMResponse:
    """Build context and call LLM to produce a reply."""
    # 1. Fetch semantic memories (best-effort)
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

    # 2. Fetch recent episode messages and convert
    recent_episode_msgs = await episode_manager.get_recent_messages(user_id, limit=20)
    recent_messages = _episode_messages_to_conversation(recent_episode_msgs)

    # 3. Assemble LLM messages via context builder
    system_prompt = get_system_prompt(user_name=user_name)
    llm_messages = get_context_builder().assemble_for_llm(
        recent_messages=recent_messages,
        semantic_memories=memories,
        query=content,
        system_prompt=system_prompt,
    )

    # 4. Call LLM
    return await get_llm_service().generate(llm_messages)


@router.message(CommandStart())
async def start(message: Message) -> None:
    """Handle /start command."""
    user_id = message.from_user.id if message.from_user else 0

    try:
        episode_manager = await get_episode_manager_service()
        await episode_manager.process_user_message(user_id=user_id, content="/start")

        response = "Привет. Я рядом 🙂\nРасскажи, как тебя зовут?"
        await episode_manager.process_assistant_message(user_id=user_id, content=response)
        await message.answer(response)

    except EpisodeManagerUnavailableError as e:
        logger.error("Episode manager unavailable for user {}: {}", user_id, e)
        await message.answer(
            "Привет. Я рядом 🙂\nРасскажи, как тебя зовут?\n\n"
            "⚠️ Примечание: в данный момент сообщения не сохраняются в базе данных."
        )
    except Exception as e:
        logger.error("Error in start handler for user {}: {}", user_id, e)
        await message.answer("Привет. Я рядом 🙂\nРасскажи, как тебя зовут?")


@router.message()
async def chat(message: Message) -> None:
    """Handle regular chat messages with LLM integration."""
    user_id = message.from_user.id if message.from_user else 0
    user_name = getattr(message.from_user, "first_name", None) if message.from_user else None
    content = message.text or ""

    if not content:
        content = _extract_message_content(message)

    try:
        episode_manager = await get_episode_manager_service()

        result = await episode_manager.process_user_message(user_id=user_id, content=content)
        logger.debug(
            "User {} message persisted to episode {} (new_episode: {}, reason: {})",
            user_id,
            result.episode.id,
            result.is_new_episode,
            result.switch_decision.reason,
        )

        # Generate LLM response
        try:
            llm_response = await _generate_llm_response(
                user_id=user_id,
                content=content,
                user_name=user_name,
                episode_manager=episode_manager,
            )
            response = llm_response.content
        except Exception as llm_exc:
            logger.error("LLM generation failed for user {}: {}", user_id, llm_exc)
            response = _LLM_FALLBACK
            llm_response = None

        # Persist assistant response (with token info when available)
        if llm_response is not None:
            await episode_manager.process_assistant_message(
                user_id=user_id,
                content=response,
                tokens_in=llm_response.tokens_in,
                tokens_out=llm_response.tokens_out,
                model=llm_response.model,
            )
        else:
            await episode_manager.process_assistant_message(
                user_id=user_id,
                content=response,
            )

        await message.answer(response)

    except EpisodeManagerUnavailableError as e:
        logger.error("Episode manager unavailable for user {}: {}", user_id, e)
        await message.answer(
            "Я тебя услышала, но сейчас не могу сохранить сообщение.\n\n"
            "⚠️ Примечание: в данный момент сообщения не сохраняются в базе данных."
        )
    except Exception as e:
        logger.error("Error in chat handler for user {}: {}", user_id, e)
        await message.answer("Я тебя услышала, но что-то пошло не так. Попробуй ещё раз.")


def _extract_message_content(message: Message) -> str:
    """Extract content from non-text messages."""
    parts = []
    if message.caption:
        parts.append(f"[Caption: {message.caption}]")
    if message.photo:
        parts.append("[Photo attached]")
    if message.document:
        parts.append(f"[Document: {message.document.file_name or 'unnamed'}]")
    if message.voice:
        parts.append("[Voice message]")
    if message.video:
        parts.append("[Video attached]")
    if message.audio:
        parts.append("[Audio attached]")
    if message.sticker:
        parts.append(f"[Sticker: {message.sticker.emoji or 'emoji'}]")
    if message.location:
        parts.append(f"[Location: {message.location.latitude}, {message.location.longitude}]")
    if message.contact:
        parts.append(f"[Contact: {message.contact.first_name or 'unnamed'}]")
    if not parts:
        parts.append("[Non-text message]")
    return " ".join(parts)
