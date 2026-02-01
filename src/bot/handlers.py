from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

router = Router()


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer("Привет. Я рядом 🙂\nРасскажи, как тебя зовут?")


@router.message()
async def chat(message: Message) -> None:
    # Placeholder: wire LangGraph + mem0 here
    await message.answer("Я тебя услышала. (Пока что это заглушка — подключим LLM + память.)")
