"""Telegram bot handlers with episode-aware message persistence."""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from loguru import logger

from bot.services.episode_manager import (
    EpisodeManager,
    get_episode_manager,
    set_episode_manager,
)

try:
    from bot.services.db_client import get_db_client

    DB_CLIENT_AVAILABLE = True
except ImportError:
    get_db_client = None  # type: ignore
    DB_CLIENT_AVAILABLE = False

router = Router()


class EpisodeManagerUnavailableError(Exception):
    """Raised when episode manager cannot be initialized.

    This is a wrapper around database initialization errors to provide
    a cleaner interface for handlers.
    """

    def __init__(
        self, message: str = "Message persistence unavailable", cause: Exception | None = None
    ) -> None:
        super().__init__(message)
        self.cause = cause


async def get_episode_manager_service() -> EpisodeManager:
    """Get or initialize episode manager service.

    Returns:
        Configured EpisodeManager instance.

    Raises:
        EpisodeManagerUnavailableError: If database initialization fails.
    """
    manager = get_episode_manager()

    # Initialize with database client if not already set
    if manager is not None and manager.db is None and DB_CLIENT_AVAILABLE:
        try:
            if get_db_client is not None:
                db_client = get_db_client()
                manager = EpisodeManager(db_client=db_client)
                set_episode_manager(manager)
        except Exception as e:
            # Log loudly and raise a clear error - do not silently fail
            logger.error(
                "Failed to initialize database client for episode manager: {}. "
                "Message persistence will be unavailable.",
                e,
            )
            raise EpisodeManagerUnavailableError(
                "Database initialization failed - messages cannot be persisted",
                cause=e,
            ) from e

    # Ensure we always return a valid EpisodeManager
    if manager is None:
        manager = EpisodeManager()
        set_episode_manager(manager)

    return manager


@router.message(CommandStart())
async def start(message: Message) -> None:
    """Handle /start command.

    Creates a new episode for the user if one doesn't exist.
    """
    user_id = message.from_user.id if message.from_user else 0

    try:
        episode_manager = await get_episode_manager_service()

        # Persist user message
        await episode_manager.process_user_message(
            user_id=user_id,
            content="/start",
        )

        response = "Привет. Я рядом 🙂\nРасскажи, как тебя зовут?"

        # Persist assistant response
        await episode_manager.process_assistant_message(
            user_id=user_id,
            content=response,
        )

        await message.answer(response)

    except EpisodeManagerUnavailableError as e:
        logger.error("Episode manager unavailable for user {}: {}", user_id, e)
        # Inform user that persistence is unavailable but still respond
        await message.answer(
            "Привет. Я рядом 🙂\nРасскажи, как тебя зовут?\n\n"
            "⚠️ Примечание: в данный момент сообщения не сохраняются в базе данных."
        )
    except Exception as e:
        logger.error("Error in start handler for user {}: {}", user_id, e)
        # Fallback response
        await message.answer("Привет. Я рядом 🙂\nРасскажи, как тебя зовут?")


@router.message()
async def chat(message: Message) -> None:
    """Handle regular chat messages.

    Persists every message with episode_id and evaluates episode switching.
    """
    user_id = message.from_user.id if message.from_user else 0
    content = message.text or ""

    if not content:
        # Handle non-text messages (photos, documents, etc.)
        content = _extract_message_content(message)

    try:
        episode_manager = await get_episode_manager_service()

        # Persist user message with episode switching logic
        result = await episode_manager.process_user_message(
            user_id=user_id,
            content=content,
        )

        logger.debug(
            "User {} message persisted to episode {} (new_episode: {}, reason: {})",
            user_id,
            result.episode.id,
            result.is_new_episode,
            result.switch_decision.reason,
        )

        # Placeholder response - replace with actual LLM integration
        response = "Я тебя услышала. (Пока что это заглушка — подключим LLM + память.)"

        # Persist assistant response
        await episode_manager.process_assistant_message(
            user_id=user_id,
            content=response,
        )

        await message.answer(response)

    except EpisodeManagerUnavailableError as e:
        logger.error("Episode manager unavailable for user {}: {}", user_id, e)
        # Inform user that persistence is unavailable but still respond
        await message.answer(
            "Я тебя услышала. (Пока что это заглушка — подключим LLM + память.)\n\n"
            "⚠️ Примечание: в данный момент сообщения не сохраняются в базе данных."
        )
    except Exception as e:
        logger.error("Error in chat handler for user {}: {}", user_id, e)
        # Fallback response without persistence
        await message.answer("Я тебя услышала. (Пока что это заглушка — подключим LLM + память.)")


def _extract_message_content(message: Message) -> str:
    """Extract content from non-text messages.

    Args:
        message: Telegram message

    Returns:
        Text representation of the message content
    """
    parts = []

    if message.caption:
        parts.append(f"[Caption: {message.caption}]")

    if message.photo:
        parts.append("[Photo attached]")

    if message.document:
        doc_name = message.document.file_name or "unnamed"
        parts.append(f"[Document: {doc_name}]")

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
