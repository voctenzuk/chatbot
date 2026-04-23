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

    redis_url: str | None = None

    # mem0 memory
    mem0_supabase_connection_string: str | None = None
    embedder_model: str = "text-embedding-3-small"

    image_base_url: str | None = None
    image_api_key: str | None = None
    image_model: str = "bytedance/seedream-4.5"

    # Langfuse observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True

    # Vision model (optional — falls back to polite refusal if not configured)
    vision_model: str | None = None
    vision_base_url: str | None = None
    max_image_size_mb: float = 5.0

    # LLM cost per 1M tokens in cents
    cost_per_1m_input: float = 0.15
    cost_per_1m_output: float = 0.60
    vision_cost_per_1m_input: float = 0.10
    vision_cost_per_1m_output: float = 0.40


settings = Settings()
