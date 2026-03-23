"""Telegram bot handlers — thin aiogram wrappers delegating to ChatPipeline."""

import base64
import mimetypes
from io import BytesIO
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, LabeledPrice, Message, PreCheckoutQuery
from loguru import logger

from bot.character import CharacterConfig
from bot.chat_pipeline import LLM_FALLBACK, ChatPipeline
from bot.config import settings

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
                    result = await img_service.generate("casual selfie, smiling warmly", user_id)
                    if result:
                        await message.answer_photo(
                            photo=BufferedInputFile(result.image_bytes, filename="hello.png"),
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
        provider_token="",  # Empty for Telegram Stars (XTR) — no third-party provider needed
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
            await _process_payment(user_id, payment, db_client, plan_slug)
            await message.answer("Подписка активирована! Спасибо 💕")
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


async def _process_payment(user_id: int, payment: Any, db_client: Any, plan_slug: str) -> None:
    """Process payment: record first, then activate. Idempotent — skips if duplicate."""
    charge_id = payment.telegram_payment_charge_id

    # Record payment BEFORE activation — ensures payment record exists for reconciliation.
    # Returns False if duplicate (ON CONFLICT DO NOTHING) or error.
    is_new = await db_client.record_payment(
        telegram_user_id=user_id,
        amount_cents=payment.total_amount,
        provider_payment_id=charge_id,
    )

    if not is_new:
        logger.warning("Duplicate payment skipped: user={} charge_id={}", user_id, charge_id)
        return

    await db_client.activate_subscription(user_id, plan_slug)

    logger.info(
        "Payment succeeded: user={} plan={} amount={} charge_id={}",
        user_id,
        plan_slug,
        payment.total_amount,
        charge_id,
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


# Catch-all handler — MUST be registered last
@router.message()
async def chat(message: Message, pipeline: ChatPipeline) -> None:
    """Handle regular chat messages — delegates to ChatPipeline."""
    user_id = message.from_user.id if message.from_user else 0
    user_name = getattr(message.from_user, "first_name", None) if message.from_user else None

    try:
        text, images = await _extract_message_content(message)
    except ValueError as exc:
        await message.answer(str(exc))
        return
    except RuntimeError as exc:
        await message.answer(str(exc))
        return

    try:
        result = await pipeline.handle_message(
            user_id=user_id,
            content=text,
            images=images,
            user_name=user_name,
        )

        # Deliver response via Telegram
        if result.image_bytes:
            if result.response_text and result.response_text.strip():
                await message.answer(result.response_text)
            for img in result.image_bytes:
                await message.answer_photo(
                    photo=BufferedInputFile(img, filename="photo.png"),
                )
        elif result.response_text and result.response_text.strip():
            await message.answer(result.response_text)

    except RuntimeError as e:
        logger.error("Episode manager runtime error for user {}: {}", user_id, e)
        await message.answer(LLM_FALLBACK)
    except Exception as e:
        logger.error("Error in chat handler for user {}: {}", user_id, e)
        await message.answer("Я тебя услышала, но что-то пошло не так. Попробуй ещё раз.")


async def _extract_message_content(message: Message) -> tuple[str, list[str] | None]:
    """Extract content and optional image data URLs from a Telegram message.

    Returns:
        (text, images) where images is a list of base64 data URLs or None.

    Raises:
        ValueError: if the photo exceeds the configured size limit.
        RuntimeError: if the photo download fails.
    """
    if message.photo:
        photo = message.photo[-1]  # largest available size

        max_bytes = int(settings.max_image_size_mb * 1024 * 1024)
        if photo.file_size and photo.file_size > max_bytes:
            raise ValueError("Ой, фотка слишком большая! Попробуй отправить поменьше 📸")

        try:
            bot = message.bot
            if bot is None:
                raise RuntimeError("Не смогла загрузить фотку, попробуй ещё раз!")
            file = await bot.get_file(photo.file_id)
            buf = BytesIO()
            await bot.download_file(file.file_path, buf)  # type: ignore[arg-type]
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError("Не смогла загрузить фотку, попробуй ещё раз!") from exc

        mime_type = "image/jpeg"  # Telegram photos are always JPEG
        if file.file_path:
            guessed = mimetypes.guess_type(file.file_path)[0]
            if guessed:
                mime_type = guessed

        b64 = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        data_url = f"data:{mime_type};base64,{b64}"
        text = message.caption or "Что на этом фото?"
        return text, [data_url]

    # Non-photo media: preserve message.caption if present
    caption = message.caption or ""

    if message.sticker:
        emoji = message.sticker.emoji or "emoji"
        return f"[Стикер: {emoji}]", None

    if message.voice:
        label = "[Голосовое сообщение]"
        return f"{caption} {label}".strip() if caption else label, None

    if message.video:
        label = "[Видео]"
        return f"{caption} {label}".strip() if caption else label, None

    if message.audio:
        label = "[Аудио]"
        return f"{caption} {label}".strip() if caption else label, None

    if message.document:
        name = message.document.file_name or "unnamed"
        label = f"[Документ: {name}]"
        return f"{caption} {label}".strip() if caption else label, None

    if message.location:
        lat = message.location.latitude
        lon = message.location.longitude
        return f"[Местоположение: {lat}, {lon}]", None

    if message.contact:
        name = message.contact.first_name or "unnamed"
        return f"[Контакт: {name}]", None

    return message.text or "", None
