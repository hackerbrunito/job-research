"""LLM Settings page.

Shows the current LLM configuration from `.env`/Settings (read-only), and
lets the user layer a *temporary* override on top for the current session.
The override lives in `st.session_state["llm_override"]` and is consumed by
the Run page when the user opts in.

Nothing on this page persists changes to `.env`. Explain to the user that
permanent changes require editing `.env`.
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from job_research import constants as C
from job_research.config import LLMConfig, get_settings
from job_research.llm_providers import build_provider
from job_research.logging_setup import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Page-local constants — no magic numbers
# --------------------------------------------------------------------------- #
SESSION_KEY_OVERRIDE: Final[str] = "llm_override"
TEST_TITLE: Final[str] = "Test"
TEST_DESCRIPTION: Final[str] = "Sample description for connectivity check."

# Provider-specific sensible model defaults used when the user switches
# provider in the override UI (so the model box auto-populates).
_PROVIDER_DEFAULT_MODEL: Final[dict[str, str]] = {
    C.PROVIDER_ANTHROPIC: C.DEFAULT_ANTHROPIC_MODEL,
    C.PROVIDER_OPENAI: C.DEFAULT_OPENAI_MODEL,
    C.PROVIDER_OPENAI_COMPATIBLE: C.DEFAULT_LOCAL_MODEL,
}

_DEFAULT_LOCAL_BASE_URL: Final[str] = "http://localhost:11434/v1"

# Number input bounds (mirror LLMConfig validators)
_MAX_TOKENS_MIN: Final[int] = 1
_MAX_TOKENS_MAX: Final[int] = 16384
_TEMPERATURE_MIN: Final[float] = 0.0
_TEMPERATURE_MAX: Final[float] = 2.0
_TEMPERATURE_STEP: Final[float] = 0.1


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sorted_providers() -> list[str]:
    """Stable ordering for UI (frozenset iteration is unordered)."""
    preferred = [C.PROVIDER_ANTHROPIC, C.PROVIDER_OPENAI, C.PROVIDER_OPENAI_COMPATIBLE]
    return [p for p in preferred if p in C.SUPPORTED_PROVIDERS]


def _effective_config(base: LLMConfig, override: dict[str, object] | None) -> LLMConfig:
    """Merge an override dict onto the base LLMConfig."""
    if not override:
        return base
    merged = base.model_dump()
    merged.update({k: v for k, v in override.items() if v is not None})
    return LLMConfig(**merged)


def _status_badge(ok: bool) -> str:
    """Plain-text status indicator — no emojis (user rule)."""
    return "[ OK ]" if ok else "[MISSING]"


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render() -> None:
    st.title("LLM Settings")
    st.caption(
        "Inspect the active LLM configuration and try temporary overrides for "
        "this session. Permanent changes require editing `.env`."
    )

    settings = get_settings()
    base_cfg = settings.llm

    # ---- Current (from .env / Settings) ---------------------------------- #
    st.subheader("Current configuration (from `.env`)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Provider:** `{base_cfg.provider}`")
        st.markdown(f"**Model:** `{base_cfg.model}`")
        st.markdown(f"**Base URL:** `{base_cfg.base_url or '—'}`")
    with col2:
        st.markdown(f"**Max tokens:** `{base_cfg.max_tokens}`")
        st.markdown(f"**Temperature:** `{base_cfg.temperature}`")
        st.markdown(f"**Timeout (s):** `{base_cfg.timeout_seconds}`")

    # ---- API key presence ------------------------------------------------- #
    st.subheader("API key presence")
    anthropic_ok = settings.anthropic_api_key is not None
    openai_ok = settings.openai_api_key is not None
    st.markdown(f"- ANTHROPIC_API_KEY: {_status_badge(anthropic_ok)}")
    st.markdown(f"- OPENAI_API_KEY: {_status_badge(openai_ok)}")
    st.caption(
        "Keys are never displayed. To add or change a key, edit `.env` and "
        "restart the Streamlit server."
    )

    st.divider()

    # ---- Session override form ------------------------------------------- #
    st.subheader("Session override (temporary)")
    existing = st.session_state.get(SESSION_KEY_OVERRIDE) or {}

    providers = _sorted_providers()
    current_provider = str(existing.get("provider") or base_cfg.provider)
    try:
        provider_index = providers.index(current_provider)
    except ValueError:
        provider_index = 0

    provider = st.selectbox(
        "Provider",
        providers,
        index=provider_index,
        help="Choose which backend to use for enrichment in this session.",
    )

    default_model = _PROVIDER_DEFAULT_MODEL.get(provider, base_cfg.model)
    current_model = str(existing.get("model") or base_cfg.model or default_model)
    # If user just switched provider, suggest that provider's default model.
    if provider != current_provider:
        current_model = default_model
    model = st.text_input("Model", value=current_model)

    base_url_value = ""
    if provider == C.PROVIDER_OPENAI_COMPATIBLE:
        current_base_url = str(
            existing.get("base_url") or base_cfg.base_url or _DEFAULT_LOCAL_BASE_URL
        )
        base_url_value = st.text_input(
            "Base URL",
            value=current_base_url,
            help="e.g. http://localhost:11434/v1 for Ollama.",
        )

    col_mt, col_temp = st.columns(2)
    with col_mt:
        max_tokens = int(
            st.number_input(
                "Max tokens",
                min_value=_MAX_TOKENS_MIN,
                max_value=_MAX_TOKENS_MAX,
                value=int(existing.get("max_tokens") or base_cfg.max_tokens),
                step=1,
            )
        )
    with col_temp:
        temperature = float(
            st.number_input(
                "Temperature",
                min_value=_TEMPERATURE_MIN,
                max_value=_TEMPERATURE_MAX,
                value=float(existing.get("temperature") or base_cfg.temperature),
                step=_TEMPERATURE_STEP,
                format="%.2f",
            )
        )

    btn_save, btn_reset, btn_test = st.columns(3)
    with btn_save:
        if st.button("Save override", width="stretch"):
            override: dict[str, object] = {
                "provider": provider,
                "model": model.strip() or default_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if provider == C.PROVIDER_OPENAI_COMPATIBLE:
                override["base_url"] = base_url_value.strip() or None
            else:
                override["base_url"] = None
            st.session_state[SESSION_KEY_OVERRIDE] = override
            log.info(
                "llm.override.saved",
                provider=provider,
                model=override["model"],
            )
            st.success("Override saved for this session.")

    with btn_reset:
        if st.button("Reset to .env defaults", width="stretch"):
            st.session_state.pop(SESSION_KEY_OVERRIDE, None)
            log.info("llm.override.cleared")
            st.success("Override cleared. Using `.env` values.")
            st.rerun()

    with btn_test:
        test_clicked = st.button("Test connection", width="stretch")

    if test_clicked:
        effective = _effective_config(
            base_cfg,
            {
                "provider": provider,
                "model": model.strip() or default_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "base_url": (
                    base_url_value.strip() or None
                    if provider == C.PROVIDER_OPENAI_COMPATIBLE
                    else None
                ),
            },
        )
        with st.status("Testing LLM connection...", expanded=True) as status:
            try:
                st.write(f"Building provider: {effective.provider} / {effective.model}")
                provider_impl = build_provider(effective, settings)
                st.write("Sending smoke-test prompt...")
                result = provider_impl.enrich(
                    title=TEST_TITLE,
                    description=TEST_DESCRIPTION,
                )
                st.write("Response received and parsed successfully.")
                status.update(label="Connection OK", state="complete")
                st.success(
                    "LLM call succeeded. "
                    f"work_mode={result.work_mode}, "
                    f"tech_skills={len(result.tech_skills)}"
                )
                log.info(
                    "llm.test.ok",
                    provider=effective.provider,
                    model=effective.model,
                )
            except Exception as exc:
                status.update(label="Connection failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
                log.warning(
                    "llm.test.failed",
                    provider=effective.provider,
                    model=effective.model,
                    error=str(exc),
                )

    # ---- Current override preview ---------------------------------------- #
    st.divider()
    st.subheader("Active session override")
    active_override = st.session_state.get(SESSION_KEY_OVERRIDE)
    if active_override:
        st.json(active_override)
    else:
        st.info("No override active. `.env` values will be used.")

    # ---- Help ------------------------------------------------------------ #
    with st.expander("How to change defaults permanently"):
        st.markdown(
            """
Edit your `.env` file at the project root and restart the server:

```
LLM__PROVIDER=anthropic
LLM__MODEL=claude-haiku-4-5-20251001
LLM__MAX_TOKENS=2048
LLM__TEMPERATURE=0.0
LLM__BASE_URL=
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

For OpenAI-compatible local servers (Ollama, LM Studio):

```
LLM__PROVIDER=openai-compatible
LLM__MODEL=llama3:8b
LLM__BASE_URL=http://localhost:11434/v1
```
            """
        )


render()
