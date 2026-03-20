"""Telegram bot handlers — thin aiogram wrappers delegating to ChatPipeline."""

from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, LabeledPrice, Message, PreCheckoutQuery
from loguru import logger

from bot.chat_pipeline import ChatPipeline
from bot.services.episode_manager import (
    EpisodeManager,
    get_episode_manager,
    set_episode_manager,
)

try:
    from bot.services.db_client import get_db_client

    DB_CLIENT_AVAILABLE = True
except ImportError:
    get_db_client = None  # type: ignore[assignment]
    DB_CLIENT_AVAILABLE = False

router = Router()

_LLM_FALLBACK = "Прости, у меня сейчас не получается ответить. Попробуй ещё раз чуть позже."


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


@router.message(CommandStart())
async def start(message: Message) -> None:
    """Handle /start command."""
    user_id = message.from_user.id if message.from_user else 0
    user_name = getattr(message.from_user, "first_name", None) if message.from_user else None

    try:
        episode_manager = await get_episode_manager_service()
        await episode_manager.process_user_message(user_id=user_id, content="/start")

        # Provision user in DB with Free plan
        if DB_CLIENT_AVAILABLE and get_db_client is not None:
            try:
                db = get_db_client()
                username = (
                    getattr(message.from_user, "username", None) if message.from_user else None
                )
                await db.provision_user(user_id, username=username, first_name=user_name)
            except Exception as e:
                logger.warning("Failed to provision user {}: {}", user_id, e)

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


@router.message(Command("upgrade"))
async def upgrade(message: Message) -> None:
    """Show subscription upgrade options via Telegram Stars."""
    await message.answer_invoice(
        title="Plus подписка",
        description="100 сообщений в день + фото. 30 дней.",
        payload="plan:plus",
        provider_token="",
        currency="XTR",
        prices=[LabeledPrice(label="Plus (30 дней)", amount=385)],
    )


@router.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery) -> None:
    """Validate payment before processing."""
    try:
        payload = query.invoice_payload or ""
        if payload.startswith("plan:"):
            await query.answer(ok=True)
        else:
            await query.answer(ok=False, error_message="Неизвестный тип оплаты")
    except Exception as e:
        logger.error("Error in pre_checkout handler for query {}: {}", query.id, e)
        await query.answer(ok=False, error_message="Ошибка при обработке платежа")


@router.message(F.successful_payment)
async def successful_payment(message: Message) -> None:
    """Handle successful Telegram Stars payment."""
    user_id = message.from_user.id if message.from_user else 0
    try:
        payment = message.successful_payment
        if not payment:
            return

        payload = payment.invoice_payload or ""
        plan_slug = payload.replace("plan:", "") if payload.startswith("plan:") else None

        if plan_slug and DB_CLIENT_AVAILABLE and get_db_client is not None:
            try:
                db = get_db_client()
                await db.activate_subscription(user_id, plan_slug)
                await message.answer("Подписка активирована! Спасибо 💕")
            except Exception as e:
                logger.error("Failed to activate subscription for user {}: {}", user_id, e)
                await message.answer(
                    "Оплата получена, но произошла ошибка. Напиши /start и попробуй снова."
                )
        else:
            await message.answer("Оплата получена! Спасибо 💕")
    except Exception as e:
        logger.error("Error in successful_payment handler for user {}: {}", user_id, e)
        await message.answer(
            "Прости, при обработке платежа что-то пошло не так. Напиши /start и попробуй снова."
        )


@router.message()
async def chat(message: Message) -> None:
    """Handle regular chat messages — delegates to ChatPipeline."""
    user_id = message.from_user.id if message.from_user else 0
    user_name = getattr(message.from_user, "first_name", None) if message.from_user else None
    content = message.text or ""

    if not content:
        content = _extract_message_content(message)

    try:
        episode_manager = await get_episode_manager_service()
        pipeline = ChatPipeline(episode_manager=episode_manager)
        result = await pipeline.handle_message(
            user_id=user_id, content=content, user_name=user_name
        )

        # Deliver response via Telegram
        if result.image_bytes is not None:
            if result.response_text and result.response_text.strip():
                await message.answer(result.response_text)
            await message.answer_photo(
                photo=BufferedInputFile(result.image_bytes, filename="photo.png"),
            )
        elif result.response_text and result.response_text.strip():
            await message.answer(result.response_text)

    except EpisodeManagerUnavailableError as e:
        logger.error("Episode manager unavailable for user {}: {}", user_id, e)
        await message.answer(
            "Я тебя услышала, но сейчас не могу сохранить сообщение.\n\n"
            "⚠️ Примечание: в данный момент сообщения не сохраняются в базе данных."
        )
    except RuntimeError as e:
        logger.error("Episode manager runtime error for user {}: {}", user_id, e)
        await message.answer(_LLM_FALLBACK)
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
