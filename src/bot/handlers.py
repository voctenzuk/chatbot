"""Telegram bot handlers — thin aiogram wrappers delegating to ChatPipeline."""

from typing import Any

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, LabeledPrice, Message, PreCheckoutQuery
from loguru import logger

from bot.character import CharacterConfig
from bot.chat_pipeline import LLM_FALLBACK, ChatPipeline

router = Router()


@router.message(CommandStart())
async def start(message: Message, pipeline: ChatPipeline) -> None:
    """Handle /start command — onboarding for new users, welcome back for returning."""
    user_id = message.from_user.id if message.from_user else 0
    user_name = getattr(message.from_user, "first_name", None) if message.from_user else None

    try:
        await pipeline.episode_manager.process_user_message(user_id=user_id, content="/start")

        # Provision user and detect new vs returning
        is_new = False
        db = pipeline.db_client
        if db is not None:
            try:
                username = (
                    getattr(message.from_user, "username", None) if message.from_user else None
                )
                result = await db.provision_user(user_id, username=username, first_name=user_name)
                if result is not None:
                    is_new = result.is_new
            except Exception as e:
                logger.warning("Failed to provision user {}: {}", user_id, e)

        character: CharacterConfig | None = pipeline.character

        if is_new and character is not None:
            # New user onboarding: greeting + photo + prompt
            await message.answer(character.greeting)

            # Free onboarding photo (bypasses pipeline rate limit)
            img_service = pipeline.image_service
            if img_service is not None:
                try:
                    photo = await img_service.generate("casual selfie, smiling warmly", user_id)
                    if photo:
                        await message.answer_photo(
                            photo=BufferedInputFile(photo, filename="hello.png"),
                        )
                except Exception as e:
                    logger.warning("Onboarding photo failed for user {}: {}", user_id, e)

            response = "Расскажи, как тебя зовут? 😊"
        elif is_new:
            response = "Привет! Я рядом 🙂\nРасскажи, как тебя зовут?"
        else:
            response = "С возвращением! 💕"

        await pipeline.episode_manager.process_assistant_message(user_id=user_id, content=response)
        await message.answer(response)

    except RuntimeError as e:
        logger.error("Episode manager runtime error in start for user {}: {}", user_id, e)
        await message.answer(
            "Привет! Я рядом 🙂\nРасскажи, как тебя зовут?\n\n"
            "⚠️ Примечание: в данный момент сообщения не сохраняются в базе данных."
        )
    except Exception as e:
        logger.error("Error in start handler for user {}: {}", user_id, e)
        await message.answer("Привет! Я рядом 🙂\nРасскажи, как тебя зовут?")


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
async def successful_payment(message: Message, db_client: Any | None = None) -> None:
    """Handle successful Telegram Stars payment."""
    user_id = message.from_user.id if message.from_user else 0
    try:
        payment = message.successful_payment
        if not payment:
            return

        payload = payment.invoice_payload or ""
        plan_slug = payload.replace("plan:", "") if payload.startswith("plan:") else None

        if plan_slug and db_client is not None:
            try:
                await db_client.activate_subscription(user_id, plan_slug)
                await message.answer("Подписка активирована! Спасибо 💕")
            except Exception as e:
                logger.error("Failed to activate subscription for user {}: {}", user_id, e)
                await message.answer(
                    "Оплата получена, но произошла ошибка. Напиши /start и попробуй снова."
                )
        elif plan_slug and db_client is None:
            logger.error("Payment received but DB unavailable for user {}", user_id)
            await message.answer(
                "Оплата получена, но произошла ошибка активации. Напиши /start и попробуй снова."
            )
        else:
            await message.answer("Оплата получена! Спасибо 💕")
    except Exception as e:
        logger.error("Error in successful_payment handler for user {}: {}", user_id, e)
        await message.answer(
            "Прости, при обработке платежа что-то пошло не так. Напиши /start и попробуй снова."
        )


@router.message(Command("stats"))
async def stats(message: Message, pipeline: ChatPipeline) -> None:
    """Show usage statistics for the current user."""
    user_id = message.from_user.id if message.from_user else 0
    db = pipeline.db_client
    if db is None:
        await message.answer("Статистика временно недоступна.")
        return
    try:
        usage = await db.get_user_usage_today(user_id)
        if not usage:
            await message.answer("Статистика пока пуста — напиши мне что-нибудь!")
            return
        msg_limit = str(usage.daily_limit) if usage.daily_limit else "∞"
        photo_limit = str(usage.photo_limit) if usage.photo_limit else "∞"
        plan_name = usage.plan_slug.title() if usage.plan_slug else "Free"
        text = (
            f"📊 Сегодня:\n"
            f"💬 Сообщений: {usage.messages_sent}/{msg_limit}\n"
            f"📷 Фото: {usage.photo_count}/{photo_limit}\n"
            f"📋 План: {plan_name}\n"
            f"📅 Дней вместе: {usage.days_together}"
        )
        await message.answer(text)
    except Exception as e:
        logger.error("Error in stats handler for user {}: {}", user_id, e)
        await message.answer("Не удалось загрузить статистику. Попробуй позже.")


@router.message()
async def chat(message: Message, pipeline: ChatPipeline) -> None:
    """Handle regular chat messages — delegates to ChatPipeline."""
    user_id = message.from_user.id if message.from_user else 0
    user_name = getattr(message.from_user, "first_name", None) if message.from_user else None
    content = message.text or ""

    if not content:
        content = _extract_message_content(message)

    try:
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

    except RuntimeError as e:
        logger.error("Episode manager runtime error for user {}: {}", user_id, e)
        await message.answer(LLM_FALLBACK)
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
