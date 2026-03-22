"""Tests for image generation service and tool-calling handler integration."""

import base64
from contextlib import nullcontext
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from bot.chat_pipeline import ChatPipeline
from bot.llm.service import LLMResponse, ToolCall


class TestSendPhotoToolSchema:
    """Tests for the SEND_PHOTO_TOOL schema."""

    def test_tool_schema_structure(self) -> None:
        from bot.media.image_service import SEND_PHOTO_TOOL

        assert SEND_PHOTO_TOOL["name"] == "send_photo"
        assert "prompt" in SEND_PHOTO_TOOL["parameters"]["properties"]
        assert "prompt" in SEND_PHOTO_TOOL["parameters"]["required"]

    def test_send_photo_tool_format(self) -> None:
        """SEND_PHOTO_TOOL should be in bare dict format for bind_tools()."""
        from bot.media.image_service import SEND_PHOTO_TOOL

        assert "name" in SEND_PHOTO_TOOL
        assert "parameters" in SEND_PHOTO_TOOL
        assert SEND_PHOTO_TOOL["name"] == "send_photo"
        # Should NOT have the wrapped OpenAI format
        assert "function" not in SEND_PHOTO_TOOL
        assert SEND_PHOTO_TOOL.get("type") != "function"


class TestImageResult:
    """Tests for ImageResult frozen dataclass."""

    def test_image_result_fields(self) -> None:
        from bot.media.image_service import ImageResult

        result = ImageResult(image_bytes=b"png-data", cost_cents=4.0, provider="openrouter")
        assert result.image_bytes == b"png-data"
        assert result.cost_cents == 4.0
        assert result.provider == "openrouter"

    def test_image_result_is_frozen(self) -> None:
        from bot.media.image_service import ImageResult

        result = ImageResult(image_bytes=b"x", cost_cents=0.0, provider="test")
        with pytest.raises(AttributeError):
            result.image_bytes = b"y"  # type: ignore[misc]


def _make_chat_completion_response(image_b64: str | None = None) -> MagicMock:
    """Build a mock chat completion response with optional base64 image."""
    response = MagicMock()
    if image_b64 is None:
        response.choices = []
    else:
        choice = MagicMock()
        choice.message.content = f"data:image/png;base64,{image_b64}"
        response.choices = [choice]
    return response


class TestImageServiceGenerate:
    """Tests for ImageService.generate() method."""

    @pytest.fixture
    def mock_openai_client(self) -> AsyncMock:
        client = AsyncMock()
        b64 = base64.b64encode(b"fake_png_data").decode()
        client.chat.completions.create = AsyncMock(return_value=_make_chat_completion_response(b64))
        return client

    @pytest.fixture
    def service(self, mock_openai_client: AsyncMock) -> Any:
        from bot.media.image_service import ImageService

        svc = ImageService.__new__(ImageService)
        svc._client = mock_openai_client
        svc._model = "bytedance/seedream-4.5"
        svc._character = None
        return svc

    @pytest.mark.asyncio
    async def test_generate_happy_path(self, service: Any, mock_openai_client: AsyncMock) -> None:
        from bot.media.image_service import ImageResult

        result = await service.generate("cute cat", user_id=123)
        assert result is not None
        assert isinstance(result, ImageResult)
        assert result.image_bytes == b"fake_png_data"
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_returns_image_result_with_cost(
        self, service: Any, mock_openai_client: AsyncMock
    ) -> None:
        from bot.media.image_service import ImageResult

        result = await service.generate("test", user_id=123)
        assert isinstance(result, ImageResult)
        assert result.cost_cents == 4.0
        assert result.provider == "bytedance/seedream-4.5"

    @pytest.mark.asyncio
    async def test_generate_empty_prompt(self, service: Any) -> None:
        result = await service.generate("", user_id=123)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_api_error_returns_none(
        self, service: Any, mock_openai_client: AsyncMock
    ) -> None:
        mock_openai_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        result = await service.generate("test", user_id=123)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_empty_response_returns_none(
        self, service: Any, mock_openai_client: AsyncMock
    ) -> None:
        """When API returns no choices, returns None without IndexError."""
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_make_chat_completion_response(None)
        )
        result = await service.generate("test", user_id=123)
        assert result is None

    @pytest.mark.asyncio
    async def test_unavailable_returns_none(self) -> None:
        from bot.media.image_service import ImageService

        svc = ImageService.__new__(ImageService)
        svc._client = None
        svc._character = None
        result = await svc.generate("test", user_id=123)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_with_reference_image(self, mock_openai_client: AsyncMock) -> None:
        """When character has reference_image_url, multimodal message is sent."""
        from bot.character import CharacterConfig
        from bot.media.image_service import ImageService

        char = CharacterConfig(
            name="T",
            personality="p",
            appearance_en="e",
            voice_style="v",
            greeting="g",
            example_messages=[],
            reference_image_url="https://example.com/ref.png",
        )
        svc = ImageService.__new__(ImageService)
        svc._client = mock_openai_client
        svc._model = "bytedance/seedream-4.5"
        svc._character = char

        await svc.generate("selfie in cafe", user_id=1)

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        content = messages[0]["content"]
        # Should be multimodal (list with image_url + text)
        assert isinstance(content, list)
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "https://example.com/ref.png"
        assert content[1]["type"] == "text"

    @pytest.mark.asyncio
    async def test_generate_fallback_on_reference_failure(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """When reference image generation fails, falls back to text-only."""
        from bot.character import CharacterConfig
        from bot.media.image_service import ImageService

        char = CharacterConfig(
            name="T",
            personality="p",
            appearance_en="Red hair",
            voice_style="v",
            greeting="g",
            example_messages=[],
            reference_image_url="https://example.com/ref.png",
        )
        svc = ImageService.__new__(ImageService)
        svc._client = mock_openai_client
        svc._model = "bytedance/seedream-4.5"
        svc._character = char

        b64 = base64.b64encode(b"fallback_data").decode()
        # First call (reference) fails, second call (text-only) succeeds
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("FLUX error"),
                _make_chat_completion_response(b64),
            ]
        )

        result = await svc.generate("selfie", user_id=1)
        assert result is not None
        assert result.image_bytes == b"fallback_data"
        assert mock_openai_client.chat.completions.create.call_count == 2


def _make_message_result(episode_id: str | None = None) -> MagicMock:
    """Build a mock MessageResult for episode_manager."""
    mock_episode = MagicMock()
    mock_episode.id = episode_id or str(uuid4())

    mock_msg = MagicMock()
    mock_msg.id = str(uuid4())
    mock_msg.episode_id = mock_episode.id

    mock_decision = MagicMock()
    mock_decision.should_switch = False
    mock_decision.reason = "same topic"
    mock_decision.confidence = 0.5
    mock_decision.trigger_type = None

    result = MagicMock()
    result.message = mock_msg
    result.episode = mock_episode
    result.is_new_episode = False
    result.switch_decision = mock_decision
    return result


class TestImageServiceAppearancePrefix:
    """Tests for character appearance_en prefix in text-only image generation."""

    @pytest.fixture
    def mock_openai_client(self) -> AsyncMock:
        client = AsyncMock()
        b64 = base64.b64encode(b"fake_png_data").decode()
        client.chat.completions.create = AsyncMock(return_value=_make_chat_completion_response(b64))
        return client

    @pytest.mark.asyncio
    async def test_generate_prepends_appearance_when_no_reference(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """Text-only path prepends appearance_en to prompt."""
        from bot.character import CharacterConfig
        from bot.media.image_service import ImageService

        char = CharacterConfig(
            name="Тест",
            personality="п",
            appearance_en="Red hair, green eyes",
            voice_style="с",
            greeting="г",
            example_messages=["м"],
        )
        svc = ImageService.__new__(ImageService)
        svc._client = mock_openai_client
        svc._model = "test-model"
        svc._character = char

        await svc.generate("sitting in a cafe", user_id=1)

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        prompt_used = messages[0]["content"]
        assert isinstance(prompt_used, str)
        assert prompt_used.startswith("Red hair, green eyes")
        assert "sitting in a cafe" in prompt_used

    @pytest.mark.asyncio
    async def test_generate_passes_prompt_unchanged_when_no_character(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """generate() passes prompt unchanged when character is None."""
        from bot.media.image_service import ImageService

        svc = ImageService.__new__(ImageService)
        svc._client = mock_openai_client
        svc._model = "test-model"
        svc._character = None

        await svc.generate("cute cat", user_id=1)

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["content"] == "cute cat"

    def test_image_service_constructor_accepts_character(self) -> None:
        """ImageService constructor accepts character parameter without error."""
        from bot.character import CharacterConfig
        from bot.media.image_service import ImageService

        char = CharacterConfig(
            name="Тест",
            personality="п",
            appearance_en="Blue eyes",
            voice_style="с",
            greeting="г",
            example_messages=["м"],
        )
        svc = ImageService.__new__(ImageService)
        svc._client = None
        svc._model = "test-model"
        svc._character = char
        assert svc._character is char


class TestHandlerToolCallIntegration:
    """Tests for tool-call-based image handling in chat handler."""

    @pytest.fixture
    def mock_episode_manager(self) -> AsyncMock:
        mgr = AsyncMock()
        mgr.process_user_message = AsyncMock(return_value=_make_message_result())
        mgr.process_assistant_message = AsyncMock(return_value=_make_message_result())
        mgr.get_recent_messages = AsyncMock(return_value=[])
        mgr.get_current_episode = AsyncMock(return_value=None)
        return mgr

    @pytest.mark.asyncio
    async def test_chat_with_tool_call_sends_image(self, mock_episode_manager: AsyncMock) -> None:
        """When LLM returns send_photo tool_call, image is generated and sent."""
        mock_msg = MagicMock()
        mock_msg.from_user = MagicMock(id=123, first_name="Test")
        mock_msg.text = "пришли фото"
        mock_msg.answer = AsyncMock()
        mock_msg.answer_photo = AsyncMock()
        mock_msg.photo = None
        mock_msg.document = None
        mock_msg.voice = None
        mock_msg.video = None
        mock_msg.audio = None
        mock_msg.sticker = None
        mock_msg.location = None
        mock_msg.contact = None
        mock_msg.caption = None

        llm_resp = LLMResponse(
            content="Вот держи!",
            model="test",
            tokens_in=10,
            tokens_out=20,
            tool_calls=[ToolCall(name="send_photo", args={"prompt": "cute selfie"}, id="t1")],
        )

        from bot.media.image_service import ImageResult

        mock_image_svc = MagicMock()
        mock_image_svc.generate = AsyncMock(
            return_value=ImageResult(
                image_bytes=b"fake_image_bytes", cost_cents=4.0, provider="test"
            )
        )

        mock_llm_svc = AsyncMock()
        mock_llm_svc.generate = AsyncMock(return_value=llm_resp)

        mock_context_builder = MagicMock()
        mock_context_builder.assemble_for_llm = MagicMock(
            return_value=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "пришли фото"},
            ]
        )

        mock_langfuse = MagicMock()
        mock_langfuse.trace = MagicMock(return_value=nullcontext())

        mock_db = AsyncMock()
        mock_db.check_rate_limit = AsyncMock(return_value=True)
        mock_db.try_consume_photo = AsyncMock(return_value=True)
        mock_db.increment_usage = AsyncMock()

        with (
            patch("bot.chat_pipeline.get_system_prompt", return_value="sys"),
            patch(
                "bot.media.image_service.SEND_PHOTO_TOOL",
                {"name": "send_photo"},
            ),
        ):
            pipeline = ChatPipeline(
                llm=mock_llm_svc,
                episode_manager=mock_episode_manager,
                context_builder=mock_context_builder,
                langfuse=mock_langfuse,
                image_service=mock_image_svc,
                db_client=mock_db,
            )

            from bot.handlers import chat

            await chat(mock_msg, pipeline=pipeline)

        mock_msg.answer_photo.assert_called_once()
        # generate() called once in tool loop; image cached and reused for delivery
        assert mock_image_svc.generate.call_count == 1
        mock_image_svc.generate.assert_called_once_with("cute selfie", 123)

    @pytest.mark.asyncio
    async def test_chat_without_tool_call_sends_text_only(
        self, mock_episode_manager: AsyncMock
    ) -> None:
        """When LLM has no tool_calls, only text is sent."""
        mock_msg = MagicMock()
        mock_msg.from_user = MagicMock(id=123, first_name="Test")
        mock_msg.text = "привет"
        mock_msg.answer = AsyncMock()
        mock_msg.answer_photo = AsyncMock()
        mock_msg.photo = None
        mock_msg.document = None
        mock_msg.voice = None
        mock_msg.video = None
        mock_msg.audio = None
        mock_msg.sticker = None
        mock_msg.location = None
        mock_msg.contact = None
        mock_msg.caption = None

        llm_resp = LLMResponse(
            content="Привет! Как дела?",
            model="test",
            tokens_in=5,
            tokens_out=10,
            tool_calls=[],
        )

        mock_llm_svc = AsyncMock()
        mock_llm_svc.generate = AsyncMock(return_value=llm_resp)

        mock_context_builder = MagicMock()
        mock_context_builder.assemble_for_llm = MagicMock(
            return_value=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "привет"},
            ]
        )

        mock_langfuse = MagicMock()
        mock_langfuse.trace = MagicMock(return_value=nullcontext())

        with patch("bot.chat_pipeline.get_system_prompt", return_value="sys"):
            pipeline = ChatPipeline(
                llm=mock_llm_svc,
                episode_manager=mock_episode_manager,
                context_builder=mock_context_builder,
                langfuse=mock_langfuse,
            )

            from bot.handlers import chat

            await chat(mock_msg, pipeline=pipeline)

        mock_msg.answer.assert_called_once_with("Привет! Как дела?")
        mock_msg.answer_photo.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_image_gen_failure_sends_text(self, mock_episode_manager: AsyncMock) -> None:
        """When image generation fails, text response is still sent."""
        mock_msg = MagicMock()
        mock_msg.from_user = MagicMock(id=123, first_name="Test")
        mock_msg.text = "фото"
        mock_msg.answer = AsyncMock()
        mock_msg.answer_photo = AsyncMock()
        mock_msg.photo = None
        mock_msg.document = None
        mock_msg.voice = None
        mock_msg.video = None
        mock_msg.audio = None
        mock_msg.sticker = None
        mock_msg.location = None
        mock_msg.contact = None
        mock_msg.caption = None

        llm_resp = LLMResponse(
            content="Вот!",
            model="test",
            tokens_in=5,
            tokens_out=10,
            tool_calls=[ToolCall(name="send_photo", args={"prompt": "test"}, id="t1")],
        )

        mock_image_svc = MagicMock()
        mock_image_svc.generate = AsyncMock(return_value=None)  # generation failed

        mock_llm_svc = AsyncMock()
        mock_llm_svc.generate = AsyncMock(return_value=llm_resp)

        mock_context_builder = MagicMock()
        mock_context_builder.assemble_for_llm = MagicMock(
            return_value=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "фото"},
            ]
        )

        mock_langfuse = MagicMock()
        mock_langfuse.trace = MagicMock(return_value=nullcontext())

        with (
            patch("bot.chat_pipeline.get_system_prompt", return_value="sys"),
            patch(
                "bot.media.image_service.SEND_PHOTO_TOOL",
                {"name": "send_photo"},
            ),
        ):
            pipeline = ChatPipeline(
                llm=mock_llm_svc,
                episode_manager=mock_episode_manager,
                context_builder=mock_context_builder,
                langfuse=mock_langfuse,
                image_service=mock_image_svc,
            )

            from bot.handlers import chat

            await chat(mock_msg, pipeline=pipeline)

        # Photo not sent, but text fallback is
        mock_msg.answer_photo.assert_not_called()
        mock_msg.answer.assert_called_once_with("Вот!")


class TestDownloadImage:
    """Tests for ImageService._download_image() helper."""

    @pytest.mark.asyncio
    async def test_download_image_happy_path(self) -> None:
        from bot.media.image_service import ImageService

        svc = ImageService.__new__(ImageService)
        svc._client = None
        svc._character = None

        with patch("bot.media.image_service.httpx.AsyncClient") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.content = b"image-data"
            mock_resp.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_resp)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.return_value = mock_client_instance

            result = await svc._download_image("https://example.com/img.png")
            assert result == b"image-data"

    @pytest.mark.asyncio
    async def test_download_image_failure_returns_none(self) -> None:
        from bot.media.image_service import ImageService

        svc = ImageService.__new__(ImageService)
        svc._client = None
        svc._character = None

        with patch("bot.media.image_service.httpx.AsyncClient") as mock_httpx:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(side_effect=httpx.HTTPError("timeout"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.return_value = mock_client_instance

            result = await svc._download_image("https://example.com/img.png")
            assert result is None


class TestExtractImageBytes:
    """Tests for ImageService._extract_image_bytes() static method."""

    def test_extract_from_data_url_string(self) -> None:
        from bot.media.image_service import ImageService

        b64 = base64.b64encode(b"test-image").decode()
        content = f"data:image/png;base64,{b64}"
        result = ImageService._extract_image_bytes(content)
        assert result == b"test-image"

    def test_extract_from_none_returns_none(self) -> None:
        from bot.media.image_service import ImageService

        assert ImageService._extract_image_bytes(None) is None

    def test_extract_from_plain_text_returns_none(self) -> None:
        from bot.media.image_service import ImageService

        assert ImageService._extract_image_bytes("just some text") is None

    def test_extract_from_list_content_blocks(self) -> None:
        """Extract image from list of content blocks (some models return this format)."""
        from bot.media.image_service import ImageService

        b64 = base64.b64encode(b"list-image").decode()
        block = MagicMock()
        block.type = "image_url"
        block.image_url = MagicMock()
        block.image_url.url = f"data:image/png;base64,{b64}"
        result = ImageService._extract_image_bytes([block])
        assert result == b"list-image"


class TestSendSpriteToolSchema:
    """Tests for SEND_SPRITE_TOOL schema."""

    def test_tool_schema_structure(self) -> None:
        from bot.media.image_service import SEND_SPRITE_TOOL

        assert SEND_SPRITE_TOOL["name"] == "send_sprite"
        assert "emotion" in SEND_SPRITE_TOOL["parameters"]["properties"]
        assert "emotion" in SEND_SPRITE_TOOL["parameters"]["required"]

    def test_enum_matches_sprite_emotions(self) -> None:
        from bot.character import SPRITE_EMOTIONS
        from bot.media.image_service import SEND_SPRITE_TOOL

        enum_list = SEND_SPRITE_TOOL["parameters"]["properties"]["emotion"]["enum"]
        assert enum_list == list(SPRITE_EMOTIONS)


class TestGetSprite:
    """Tests for ImageService.get_sprite() method."""

    def _make_service_with_sprites(self) -> Any:
        from bot.character import CharacterConfig
        from bot.media.image_service import ImageService

        char = CharacterConfig(
            name="T",
            personality="p",
            appearance_en="e",
            voice_style="v",
            greeting="g",
            example_messages=[],
            sprite_urls={
                "smile": "https://storage.example.com/smile.png",
                "sad": "https://storage.example.com/sad.png",
            },
        )
        svc = ImageService.__new__(ImageService)
        svc._client = None
        svc._model = "test"
        svc._character = char
        svc._sprite_cache = {}
        return svc

    @pytest.mark.asyncio
    async def test_get_sprite_happy_path(self) -> None:
        svc = self._make_service_with_sprites()
        with patch.object(svc, "_download_image", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = b"smile-png"
            result = await svc.get_sprite("smile")
            assert result == b"smile-png"
            mock_dl.assert_called_once_with("https://storage.example.com/smile.png")

    @pytest.mark.asyncio
    async def test_get_sprite_cache_hit(self) -> None:
        svc = self._make_service_with_sprites()
        with patch.object(svc, "_download_image", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = b"smile-png"
            await svc.get_sprite("smile")
            result2 = await svc.get_sprite("smile")
            assert result2 == b"smile-png"
            assert mock_dl.call_count == 1  # Only downloaded once

    @pytest.mark.asyncio
    async def test_get_sprite_unknown_emotion_returns_none(self) -> None:
        svc = self._make_service_with_sprites()
        result = await svc.get_sprite("angry")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_sprite_no_character_returns_none(self) -> None:
        from bot.media.image_service import ImageService

        svc = ImageService.__new__(ImageService)
        svc._client = None
        svc._model = "test"
        svc._character = None
        svc._sprite_cache = {}
        result = await svc.get_sprite("smile")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_sprite_no_sprite_urls_returns_none(self) -> None:
        from bot.character import CharacterConfig
        from bot.media.image_service import ImageService

        char = CharacterConfig(
            name="T",
            personality="p",
            appearance_en="e",
            voice_style="v",
            greeting="g",
            example_messages=[],
        )
        svc = ImageService.__new__(ImageService)
        svc._client = None
        svc._model = "test"
        svc._character = char
        svc._sprite_cache = {}
        result = await svc.get_sprite("smile")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_sprite_download_failure_returns_none(self) -> None:
        svc = self._make_service_with_sprites()
        with patch.object(svc, "_download_image", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = None  # download failed
            result = await svc.get_sprite("smile")
            assert result is None
            # Should NOT cache the failure
            assert "smile" not in svc._sprite_cache
