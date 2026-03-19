"""Telegram bot handlers with episode-aware message persistence and LLM integration."""

from __future__ import annotations

import asyncio
from datetime import datetime

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import LabeledPrice, Message, PreCheckoutQuery
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

# Counter for scheduling periodic cognify() calls
_memory_write_counts: dict[int, int] = {}
_COGNIFY_EVERY_N_WRITES: int = 10


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


async def _run_cognify_background() -> None:
    """Run cognify in background to build knowledge graph. Fire-and-forget."""
    try:
        if MEMORY_SERVICE_AVAILABLE and get_memory_service is not None:
            await get_memory_service().cognify()
            logger.info("Background cognify completed successfully")
    except Exception as exc:
        logger.warning("Background cognify failed: {}", exc)


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
        currency="XTR",
        prices=[LabeledPrice(label="Plus (30 дней)", amount=385)],
    )


@router.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery) -> None:
    """Validate payment before processing."""
    payload = query.invoice_payload or ""
    if payload.startswith("plan:"):
        await query.answer(ok=True)
    else:
        await query.answer(ok=False, error_message="Неизвестный тип оплаты")


@router.message(F.successful_payment)
async def successful_payment(message: Message) -> None:
    """Handle successful Telegram Stars payment."""
    user_id = message.from_user.id if message.from_user else 0
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

        # Check rate limit (fail open on errors)
        if DB_CLIENT_AVAILABLE and get_db_client is not None:
            try:
                db = get_db_client()
                allowed = await db.check_rate_limit(user_id)
                if not allowed:
                    await message.answer(
                        "Сегодня лимит сообщений исчерпан 😔\n"
                        "Напиши /upgrade чтобы увеличить лимит."
                    )
                    return
            except Exception as exc:
                logger.warning("Rate limit check failed for user {}, allowing: {}", user_id, exc)

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

        # Track usage (fire-and-forget)
        if DB_CLIENT_AVAILABLE and get_db_client is not None and llm_response is not None:
            try:
                from bot.services.llm_service import estimate_cost_cents

                db = get_db_client()
                cost = estimate_cost_cents(
                    llm_response.model, llm_response.tokens_in, llm_response.tokens_out
                )
                await db.increment_usage(
                    user_id,
                    msg_count=1,
                    tokens_in=llm_response.tokens_in,
                    tokens_out=llm_response.tokens_out,
                    cost_cents=cost,
                )
            except Exception as exc:
                logger.warning("Usage tracking failed for user {}: {}", user_id, exc)

        # Write conversation to long-term memory (fire-and-forget)
        if MEMORY_SERVICE_AVAILABLE and get_memory_service is not None and llm_response is not None:
            try:
                mem_service = get_memory_service()
                memory_content = f"Пользователь: {content}\nПодруга: {response}"
                await mem_service.write_factual(
                    content=memory_content,
                    user_id=user_id,
                )

                # Schedule cognify() periodically to build knowledge graph
                _memory_write_counts[user_id] = _memory_write_counts.get(user_id, 0) + 1
                if _memory_write_counts[user_id] >= _COGNIFY_EVERY_N_WRITES:
                    _memory_write_counts[user_id] = 0
                    asyncio.create_task(_run_cognify_background())
            except Exception as exc:
                logger.warning("Memory write failed for user {}: {}", user_id, exc)

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
