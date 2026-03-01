from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    model_name: str = Field("claude-sonnet-4-6", description="Claude model to use")

    # Web search
    tavily_api_key: str = Field(..., description="Tavily API key")
    max_search_results: int = Field(8, description="Max results per Tavily query")
    max_sources_to_extract: int = Field(5, description="Max sources to extract/summarize")

    # Database
    database_url: str = Field(..., description="Async SQLAlchemy database URL")
    sync_database_url: str = Field(..., description="Sync database URL for Alembic + checkpointer")

    # Retry
    max_retries: int = Field(3, description="Max retries for tool calls")

    # Cost tracking (USD per 1K tokens)
    cost_input_per_1k: float = Field(0.003, description="Input token cost per 1K")
    cost_output_per_1k: float = Field(0.015, description="Output token cost per 1K")

    # Logging
    log_level: str = Field("INFO", description="Logging level")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
