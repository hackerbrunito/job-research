"""Tests for AnthropicProvider.enrich() and _OpenAIStyleProvider.enrich()."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import openai
import pytest
from pytest_mock import MockerFixture

from job_research import constants as C
from job_research.config import LLMConfig, Settings
from job_research.llm_providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    build_provider,
)
from job_research.schemas import JobEnrichment, LocationExtraction, SalaryExtraction


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_sleep(mocker: MockerFixture) -> None:
    """Disable tenacity's backoff sleeps across all tests in this module."""
    mocker.patch("tenacity.nap.time.sleep", return_value=None)


def _valid_enrichment_dict() -> dict:
    return {
        "tech_skills": ["python", "sql"],
        "soft_skills": ["collaborative"],
        "location": {"city": "NYC", "country": "USA", "country_code": "US"},
        "work_mode": C.WORK_MODE_REMOTE,
        "salary": {
            "min_amount": 100000,
            "max_amount": 150000,
            "currency": "USD",
            "period": C.SALARY_PERIOD_YEARLY,
        },
    }


def _anthropic_http_error(cls: type, status: int) -> anthropic.APIStatusError:
    """Build a real Anthropic SDK error instance."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req)
    return cls("simulated", response=resp, body=None)


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class TestAnthropicEnrich:
    def _cfg(self) -> LLMConfig:
        return LLMConfig(
            provider=C.PROVIDER_ANTHROPIC,
            model=C.DEFAULT_ANTHROPIC_MODEL,
            max_retries=2,
        )

    def test_anthropic_enrich_happy_path(self, mocker: MockerFixture) -> None:
        fake_client = MagicMock()
        tool_block = SimpleNamespace(type="tool_use", input=_valid_enrichment_dict())
        fake_client.messages.create.return_value = SimpleNamespace(content=[tool_block])
        mocker.patch(
            "job_research.llm_providers.anthropic.Anthropic", return_value=fake_client
        )

        provider = AnthropicProvider(self._cfg(), api_key="sk-ant-test")
        result = provider.enrich(title="Python Dev", description="Build APIs")

        assert isinstance(result, JobEnrichment)
        assert result.tech_skills == ["python", "sql"]
        assert result.location.country_code == "US"
        assert fake_client.messages.create.call_count == 1

    def test_anthropic_enrich_raises_when_no_tool_block(
        self, mocker: MockerFixture
    ) -> None:
        fake_client = MagicMock()
        fake_client.messages.create.return_value = SimpleNamespace(content=[])
        mocker.patch(
            "job_research.llm_providers.anthropic.Anthropic", return_value=fake_client
        )

        provider = AnthropicProvider(self._cfg(), api_key="sk-ant-test")
        with pytest.raises(ValueError, match="tool_use"):
            provider.enrich(title="x", description="y")

    def test_anthropic_enrich_retries_on_rate_limit(
        self, mocker: MockerFixture
    ) -> None:
        fake_client = MagicMock()
        tool_block = SimpleNamespace(type="tool_use", input=_valid_enrichment_dict())
        fake_client.messages.create.side_effect = [
            _anthropic_http_error(anthropic.RateLimitError, 429),
            SimpleNamespace(content=[tool_block]),
        ]
        mocker.patch(
            "job_research.llm_providers.anthropic.Anthropic", return_value=fake_client
        )

        provider = AnthropicProvider(self._cfg(), api_key="sk-ant-test")
        result = provider.enrich(title="x", description="y")

        assert isinstance(result, JobEnrichment)
        assert fake_client.messages.create.call_count == 2

    def test_anthropic_enrich_does_not_retry_on_400(
        self, mocker: MockerFixture
    ) -> None:
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = _anthropic_http_error(
            anthropic.BadRequestError, 400
        )
        mocker.patch(
            "job_research.llm_providers.anthropic.Anthropic", return_value=fake_client
        )

        provider = AnthropicProvider(self._cfg(), api_key="sk-ant-test")
        with pytest.raises(anthropic.BadRequestError):
            provider.enrich(title="x", description="y")

        assert fake_client.messages.create.call_count == 1


# --------------------------------------------------------------------------- #
# OpenAI / OpenAI-compatible
# --------------------------------------------------------------------------- #
class TestOpenAIEnrich:
    def _cfg(self, provider: str = C.PROVIDER_OPENAI) -> LLMConfig:
        return LLMConfig(
            provider=provider,
            model=C.DEFAULT_OPENAI_MODEL,
            max_retries=2,
        )

    def test_openai_enrich_happy_path(self, mocker: MockerFixture) -> None:
        parsed_obj = JobEnrichment(
            tech_skills=["python"],
            soft_skills=[],
            location=LocationExtraction(country_code="US"),
            work_mode=C.WORK_MODE_REMOTE,
            salary=SalaryExtraction(currency="USD", period=C.SALARY_PERIOD_YEARLY),
        )

        fake_client = MagicMock()
        fake_message = SimpleNamespace(parsed=parsed_obj, refusal=None)
        fake_choice = SimpleNamespace(message=fake_message)
        fake_client.beta.chat.completions.parse.return_value = SimpleNamespace(
            choices=[fake_choice]
        )
        mocker.patch(
            "job_research.llm_providers.openai.OpenAI", return_value=fake_client
        )

        provider = OpenAIProvider(self._cfg(), api_key="sk-test")
        result = provider.enrich(title="x", description="y")

        assert result is parsed_obj
        assert fake_client.beta.chat.completions.parse.call_count == 1

    def test_openai_compatible_sends_dummy_key_when_absent(
        self, mocker: MockerFixture
    ) -> None:
        """When no openai_api_key set, factory must pass a non-empty placeholder."""
        openai_mock = mocker.patch(
            "job_research.llm_providers.openai.OpenAI", return_value=MagicMock()
        )

        cfg = LLMConfig(
            provider=C.PROVIDER_OPENAI_COMPATIBLE,
            model=C.DEFAULT_LOCAL_MODEL,
            base_url="http://localhost:11434/v1",
        )
        settings = Settings(anthropic_api_key=None, openai_api_key=None)

        provider = build_provider(cfg, settings)
        assert isinstance(provider, OpenAICompatibleProvider)

        # Verify the client was constructed with a non-empty api_key + base_url.
        assert openai_mock.call_count == 1
        kwargs = openai_mock.call_args.kwargs
        assert kwargs["api_key"]  # non-empty placeholder
        assert kwargs["api_key"] != ""
        assert kwargs["base_url"] == "http://localhost:11434/v1"


__all__ = ["openai"]  # silence unused import in some configs
