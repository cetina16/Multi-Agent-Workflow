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

    # LLM (Groq — free tier)
    groq_api_key: str = Field(..., description="Groq API key (free at console.groq.com)")
    model_name: str = Field("llama-3.3-70b-versatile", description="Groq model to use")

    # Web search (DuckDuckGo — no API key required)
    max_search_results: int = Field(8, description="Max results per search query")
    max_sources_to_extract: int = Field(5, description="Max sources to extract/summarize")

    # Database (SQLite — no server required)
    sqlite_path: str = Field("research.db", description="Path to SQLite database file")

    # Retry
    max_retries: int = Field(3, description="Max retries for tool calls")

    # Cost tracking (Groq free tier = $0, but tracking still works)
    cost_input_per_1k: float = Field(0.0, description="Input token cost per 1K")
    cost_output_per_1k: float = Field(0.0, description="Output token cost per 1K")

    # Logging
    log_level: str = Field("INFO", description="Logging level")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
