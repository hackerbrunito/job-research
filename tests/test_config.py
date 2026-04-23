"""Tests for config.py validators (provider, log_level, base_url)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from job_research.config import LLMConfig, Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Ensure lru_cached get_settings does not leak between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Provider validator
# --------------------------------------------------------------------------- #
def test_invalid_provider_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(provider="grok")


# --------------------------------------------------------------------------- #
# log_level validator
# --------------------------------------------------------------------------- #
def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValidationError):
        Settings()


# --------------------------------------------------------------------------- #
# base_url validator
# --------------------------------------------------------------------------- #
def test_base_url_rejects_private_ip() -> None:
    """Cloud metadata address — SSRF risk."""
    with pytest.raises(ValidationError):
        LLMConfig(base_url="http://169.254.169.254/")


def test_base_url_rejects_private_rfc1918() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(base_url="http://10.0.0.5/")


def test_base_url_allows_localhost() -> None:
    cfg = LLMConfig(base_url="http://localhost:11434/v1")
    assert cfg.base_url is not None


def test_base_url_allows_loopback_ip() -> None:
    cfg = LLMConfig(base_url="http://127.0.0.1:11434/v1")
    assert cfg.base_url is not None


def test_base_url_allows_public_dns() -> None:
    cfg = LLMConfig(base_url="https://api.example.com/v1")
    assert cfg.base_url is not None


def test_base_url_rejects_non_http_scheme() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(base_url="file:///etc/passwd")
