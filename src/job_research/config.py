"""Application settings via pydantic-settings.

All configuration flows through a single `Settings` object. Nested groups
use the `__` delimiter in env vars (e.g. `LLM__MODEL`, `SCRAPING__MAX_RETRIES`).

Import pattern:
    from job_research.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from job_research import constants as C


# --------------------------------------------------------------------------- #
# Nested config groups
# --------------------------------------------------------------------------- #
class LLMConfig(BaseModel):
    """LLM provider + model configuration.

    For `openai-compatible` providers (Ollama, LM Studio, Open WebUI),
    set `base_url` and leave `api_key_env` unused.
    """

    provider: str = C.PROVIDER_ANTHROPIC
    model: str = C.DEFAULT_ANTHROPIC_MODEL
    max_tokens: int = Field(default=C.DEFAULT_LLM_MAX_TOKENS, gt=0, le=16384)
    temperature: float = Field(default=C.DEFAULT_LLM_TEMPERATURE, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=C.DEFAULT_LLM_TIMEOUT_SECONDS, gt=0)
    max_retries: int = Field(default=C.DEFAULT_LLM_MAX_RETRIES, ge=0)
    base_url: str | None = None

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in C.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"provider must be one of {sorted(C.SUPPORTED_PROVIDERS)}, got {v!r}"
            )
        return v


class ScrapingConfig(BaseModel):
    max_results_per_site: int = Field(default=C.DEFAULT_RESULTS_PER_SITE, gt=0, le=1000)
    hours_old: int = Field(default=C.DEFAULT_HOURS_OLD, gt=0)
    request_delay_seconds: float = Field(
        default=C.DEFAULT_REQUEST_DELAY_SECONDS, ge=0.0
    )
    max_retries: int = Field(default=C.DEFAULT_SCRAPING_MAX_RETRIES, ge=0)
    timeout_seconds: int = Field(default=C.DEFAULT_SCRAPING_TIMEOUT_SECONDS, gt=0)
    linkedin_fetch_description: bool = True
    proxies: str | None = None  # comma-separated list

    def proxy_list(self) -> list[str]:
        return (
            [p.strip() for p in self.proxies.split(",") if p.strip()]
            if self.proxies
            else []
        )


class DatabaseConfig(BaseModel):
    path: Path = Path(C.DEFAULT_DB_PATH)
    read_only: bool = False

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(v) if isinstance(v, str) else v  # type: ignore[return-value]


class DashboardConfig(BaseModel):
    host: str = C.DEFAULT_DASHBOARD_HOST
    port: int = Field(default=C.DEFAULT_DASHBOARD_PORT, gt=0, le=65535)


# --------------------------------------------------------------------------- #
# Root settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Root application settings, loaded from env + `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # General
    debug: bool = False
    log_level: str = "INFO"

    # API keys — SecretStr masks in repr/logs. Both optional so the app can
    # start without one if the user intends to use a local OpenAI-compatible
    # server; the LLM provider factory enforces presence when needed.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # Nested groups
    llm: LLMConfig = LLMConfig()
    scraping: ScrapingConfig = ScrapingConfig()
    database: DatabaseConfig = DatabaseConfig()
    dashboard: DashboardConfig = DashboardConfig()

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, v: str) -> str:
        v = v.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (loads env/.env once per process)."""
    return Settings()
