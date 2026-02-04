from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: str | None = None

    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "kimi-k2p5"

    mem0_api_key: str | None = None
    mem0_project_id: str | None = None

    redis_url: str | None = None

    image_provider: str | None = None
    openai_api_key: str | None = None
    openai_image_model: str = "gpt-image-1"


settings = Settings()
