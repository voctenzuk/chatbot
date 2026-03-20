from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # None default allows tests and import-time usage without a real token;
    # app.py raises RuntimeError at startup if still None.
    telegram_bot_token: str | None = None

    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "kimi-k2p5"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    cognee_vector_db_provider: str = "lancedb"
    cognee_graph_db_provider: str = "kuzu"

    redis_url: str | None = None

    image_base_url: str | None = None
    image_api_key: str | None = None
    image_model: str = "gpt-image-1"

    # Langfuse observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True


settings = Settings()
