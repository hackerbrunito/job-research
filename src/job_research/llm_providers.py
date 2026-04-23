"""LLM provider abstraction for enrichment.

Three backends share a single `LLMProvider` protocol:

- `AnthropicProvider`        — Claude via tool-use structured output.
- `OpenAIProvider`           — OpenAI via `beta.chat.completions.parse()`.
- `OpenAICompatibleProvider` — Ollama / LM Studio / Open WebUI via the
  same `parse()` method but with a custom `base_url` and a tolerant key.

All three emit a validated `JobEnrichment`. Transient failures (timeouts,
rate limits, 5xx) are retried with exponential backoff via `tenacity`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import anthropic
import openai
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from job_research import constants as C
from job_research.config import LLMConfig, Settings
from job_research.logging_setup import get_logger
from job_research.schemas import JobEnrichment

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT: str = (
    "You extract structured data from job postings. "
    "Return: tech_skills (lowercase tools/languages/frameworks/platforms, "
    "deduped), soft_skills (lowercase adjective-form, e.g. 'collaborative', "
    "'analytical', deduped), location (city, country, country_code as ISO "
    "3166-1 alpha-2 two-letter uppercase), work_mode (one of remote | hybrid "
    "| on-site), salary (min_amount, max_amount as numbers, currency as "
    "3-letter ISO 4217 uppercase, period as one of yearly | monthly | "
    "hourly). Use null for anything not explicitly stated. Do not guess."
)

TOOL_NAME: str = "emit_enrichment"


def _user_content(*, title: str, description: str) -> str:
    """Format a single job posting as a user message."""
    title = (title or "").strip() or "(untitled)"
    description = (description or "").strip() or "(no description)"
    return f"Job title: {title}\n\nJob description:\n{description}"


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class LLMProvider(Protocol):
    """Single-method interface every provider implements."""

    provider_name: str
    model_name: str

    def enrich(self, *, title: str, description: str) -> JobEnrichment: ...


# --------------------------------------------------------------------------- #
# Retry helpers
# --------------------------------------------------------------------------- #
_ANTHROPIC_RETRY = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,
)

_OPENAI_RETRY = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def _build_retry(exc_types: tuple[type[BaseException], ...], max_retries: int):
    """Return a tenacity decorator matching the config."""
    # stop_after_attempt counts total attempts, so max_retries + 1.
    return retry(
        reraise=True,
        retry=retry_if_exception_type(exc_types),
        stop=stop_after_attempt(max(1, max_retries + 1)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class AnthropicProvider:
    """Claude provider using the tool-use structured-output pattern."""

    provider_name = C.PROVIDER_ANTHROPIC

    def __init__(self, cfg: LLMConfig, api_key: str) -> None:
        self._cfg = cfg
        self.model_name = cfg.model
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=float(cfg.timeout_seconds),
            max_retries=0,  # we handle retries via tenacity
        )
        self._tool = {
            "name": TOOL_NAME,
            "description": "Emit the structured enrichment for the job posting.",
            "input_schema": JobEnrichment.model_json_schema(),
        }

    def enrich(self, *, title: str, description: str) -> JobEnrichment:
        @_build_retry(_ANTHROPIC_RETRY, self._cfg.max_retries)
        def _call() -> JobEnrichment:
            response = self._client.messages.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                system=SYSTEM_PROMPT,
                tools=[self._tool],
                tool_choice={"type": "tool", "name": TOOL_NAME},
                messages=[
                    {
                        "role": "user",
                        "content": _user_content(title=title, description=description),
                    }
                ],
            )
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_blocks:
                raise ValueError("Anthropic response contained no tool_use block")
            return JobEnrichment.model_validate(tool_blocks[0].input)

        return _call()


# --------------------------------------------------------------------------- #
# OpenAI (and OpenAI-compatible)
# --------------------------------------------------------------------------- #
class _OpenAIStyleProvider:
    """Shared behaviour for OpenAI and OpenAI-compatible backends."""

    provider_name: str = C.PROVIDER_OPENAI

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self._cfg = cfg
        self.model_name = cfg.model
        client_kwargs: dict[str, object] = {
            "api_key": api_key,
            "timeout": float(cfg.timeout_seconds),
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)  # type: ignore[arg-type]

    def enrich(self, *, title: str, description: str) -> JobEnrichment:
        @_build_retry(_OPENAI_RETRY, self._cfg.max_retries)
        def _call() -> JobEnrichment:
            completion = self._client.beta.chat.completions.parse(
                model=self._cfg.model,
                temperature=self._cfg.temperature,
                max_completion_tokens=self._cfg.max_tokens,
                response_format=JobEnrichment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _user_content(title=title, description=description),
                    },
                ],
            )
            choice = completion.choices[0]
            if getattr(choice.message, "refusal", None):
                raise ValueError(
                    f"Model refused structured extraction: {choice.message.refusal}"
                )
            parsed = choice.message.parsed
            if parsed is None:
                raise ValueError("OpenAI response contained no parsed payload")
            # parsed is already a JobEnrichment, but re-validate defensively
            # to apply our cleanup validators if the caller mutated anything.
            if isinstance(parsed, JobEnrichment):
                return parsed
            return JobEnrichment.model_validate(parsed)

        return _call()


class OpenAIProvider(_OpenAIStyleProvider):
    provider_name = C.PROVIDER_OPENAI


class OpenAICompatibleProvider(_OpenAIStyleProvider):
    """Works with Ollama / LM Studio / Open WebUI via `base_url`."""

    provider_name = C.PROVIDER_OPENAI_COMPATIBLE


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_provider(cfg: LLMConfig, settings: Settings) -> LLMProvider:
    """Instantiate the provider described by `cfg`.

    Raises:
        ValueError: if required credentials are missing or config is invalid.
    """
    provider = cfg.provider

    if provider == C.PROVIDER_ANTHROPIC:
        if settings.anthropic_api_key is None:
            raise ValueError("provider=anthropic requires ANTHROPIC_API_KEY to be set")
        return AnthropicProvider(
            cfg, api_key=settings.anthropic_api_key.get_secret_value()
        )

    if provider == C.PROVIDER_OPENAI:
        if settings.openai_api_key is None:
            raise ValueError("provider=openai requires OPENAI_API_KEY to be set")
        return OpenAIProvider(cfg, api_key=settings.openai_api_key.get_secret_value())

    if provider == C.PROVIDER_OPENAI_COMPATIBLE:
        if not cfg.base_url:
            raise ValueError(
                "provider=openai-compatible requires `llm.base_url` to be set"
            )
        # The SDK insists on a non-empty api_key string even when the remote
        # server ignores it; pass a sentinel when none is configured.
        key = (
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key is not None
            else "sk-not-needed"
        )
        return OpenAICompatibleProvider(cfg, api_key=key, base_url=cfg.base_url)

    # Shouldn't happen — LLMConfig validator rejects unknown providers — but
    # guard anyway in case a new provider is added to constants without a
    # matching factory branch.
    raise ValueError(f"unsupported provider: {provider!r}")


__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_NAME",
    "AnthropicProvider",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "ValidationError",
    "build_provider",
]
