"""Run Pipeline page.

Shows the saved search config (from the Search page, persisted by Agent A's
`app.common.get_search_config`) and triggers the Prefect flow on demand.

Design rules honoured:
- Pipeline runs ONLY on button click, never on page load.
- The run button is disabled while a run is in progress.
- After a successful or failed run, all `st.cache_data` is cleared so that
  Results and History pages reflect the new state.
"""

from __future__ import annotations

from typing import Any, Final

import streamlit as st

from job_research import constants as C
from job_research.config import LLMConfig, Settings
from job_research.logging_setup import get_logger
from job_research.pipeline import PipelineSummary, job_research_pipeline

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Page-local constants
# --------------------------------------------------------------------------- #
SESSION_KEY_RUNNING: Final[str] = "pipeline_running"
SESSION_KEY_LAST_SUMMARY: Final[str] = "pipeline_last_summary"
SESSION_KEY_LLM_OVERRIDE: Final[str] = "llm_override"

_ENRICH_LIMIT_MIN: Final[int] = 1
_ENRICH_LIMIT_MAX: Final[int] = 10_000
_ENRICH_LIMIT_DEFAULT: Final[int] = 50


# --------------------------------------------------------------------------- #
# Search config retrieval (with graceful fallback if Agent A's common.py
# is not yet present)
# --------------------------------------------------------------------------- #
def _load_search_config() -> dict[str, Any] | None:
    """Return persisted search config or None if nothing saved yet."""
    try:
        # Agent A owns app.common; import lazily so this page still loads
        # even if the module doesn't exist yet.
        from job_research.app import (
            common as app_common,  # type: ignore[import-not-found]
        )
        from job_research.database import connect

        with connect(read_only=True) as con:
            cfg = app_common.get_search_config(con)
        if cfg is None:
            return None
        # Be defensive: accept either a dict-like or an object with attrs.
        if isinstance(cfg, dict):
            return cfg
        return {
            "keywords": getattr(cfg, "keywords", []),
            "locations": getattr(cfg, "locations", []),
            "sites": getattr(cfg, "sites", list(C.DEFAULT_SITES)),
        }
    except Exception as exc:
        log.debug("run.search_config.unavailable", error=str(exc))
        return None


# --------------------------------------------------------------------------- #
# LLM override application
# --------------------------------------------------------------------------- #
def _apply_llm_override(settings: Settings, override: dict[str, Any]) -> Settings:
    """Return a Settings copy with LLMConfig merged from the override.

    We don't mutate the cached Settings instance. Instead, we build a new
    LLMConfig and replace it on a shallow copy. `get_settings` everywhere
    else still returns the cached original — the pipeline reads settings
    via `get_settings()` inside tasks, so for the override to take effect
    we monkey-patch the cache for the duration of the run.
    """
    merged = settings.llm.model_dump()
    merged.update({k: v for k, v in override.items() if v is not None})
    new_llm = LLMConfig(**merged)
    new_settings = settings.model_copy(update={"llm": new_llm})
    return new_settings


def _run_with_override(
    *,
    keywords: list[str],
    locations: list[str] | None,
    sites: list[str] | None,
    enrich_limit: int | None,
    override: dict[str, Any] | None,
) -> PipelineSummary:
    """Invoke the flow, optionally swapping the cached Settings for the run."""
    from job_research import config as cfg_module

    original = cfg_module.get_settings()
    patched = _apply_llm_override(original, override) if override else None
    _orig_fn = cfg_module.get_settings
    try:
        if patched is not None:
            # Monkey-patch the module-level `get_settings` so pipeline tasks
            # (which call `get_settings()` internally) see the overridden
            # LLMConfig. Restored in `finally`.
            def _patched_get_settings() -> Settings:
                return patched

            cfg_module.get_settings = _patched_get_settings  # type: ignore[assignment]

        summary = job_research_pipeline(
            keywords=keywords,
            locations=locations,
            sites=sites,
            enrich_limit=enrich_limit,
        )
        return summary
    finally:
        if patched is not None:
            cfg_module.get_settings = _orig_fn  # type: ignore[assignment]
            cfg_module.get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render() -> None:
    st.title("Run Pipeline")
    st.caption("Trigger the scrape -> enrich -> transform pipeline on demand.")

    # ---- Search config summary ------------------------------------------- #
    st.subheader("Search configuration")
    cfg = _load_search_config()
    if cfg is None:
        st.warning(
            "No saved search configuration found. Open the **Search** page to "
            "set keywords, locations, and sites before running the pipeline."
        )
        return

    keywords: list[str] = list(cfg.get("keywords") or [])
    locations: list[str] = list(cfg.get("locations") or [])
    sites: list[str] = list(cfg.get("sites") or list(C.DEFAULT_SITES))

    if not keywords:
        st.warning(
            "Your saved search has no keywords. Add at least one keyword on "
            "the **Search** page."
        )
        return

    col_k, col_l, col_s = st.columns(3)
    with col_k:
        st.markdown("**Keywords**")
        st.write(", ".join(keywords))
    with col_l:
        st.markdown("**Locations**")
        st.write(", ".join(locations) if locations else "(none — global)")
    with col_s:
        st.markdown("**Sites**")
        st.write(", ".join(sites))

    # keywords x max(1, locations) scrape requests
    request_count = len(keywords) * max(1, len(locations))
    st.info(f"Planned scrape requests: **{request_count}** (keywords x locations).")

    st.divider()

    # ---- Run options ----------------------------------------------------- #
    st.subheader("Run options")

    limit_enabled = st.checkbox(
        "Limit enrichment (for testing)",
        value=False,
        help="Cap the number of rows sent to the LLM. Useful for dev/debug.",
    )
    enrich_limit: int | None = None
    if limit_enabled:
        enrich_limit = int(
            st.number_input(
                "Enrich limit",
                min_value=_ENRICH_LIMIT_MIN,
                max_value=_ENRICH_LIMIT_MAX,
                value=_ENRICH_LIMIT_DEFAULT,
                step=1,
            )
        )

    override = st.session_state.get(SESSION_KEY_LLM_OVERRIDE)
    use_override = False
    if override:
        use_override = st.checkbox(
            "Use LLM override from the LLM Settings page",
            value=True,
            help=(
                "Applies the temporary overrides set on the LLM page "
                "(provider, model, etc.) for this run only."
            ),
        )
        if use_override:
            st.caption(
                f"Override active — provider: `{override.get('provider')}`, "
                f"model: `{override.get('model')}`"
            )
    else:
        st.caption("No LLM override active — using `.env` values.")

    st.divider()

    # ---- Trigger --------------------------------------------------------- #
    running = bool(st.session_state.get(SESSION_KEY_RUNNING, False))
    run_clicked = st.button(
        "Run pipeline",
        type="primary",
        disabled=running,
        use_container_width=False,
    )

    if run_clicked:
        st.session_state[SESSION_KEY_RUNNING] = True
        with st.status("Running pipeline...", expanded=True) as status:
            try:
                st.write(
                    f"Scraping {len(keywords)} keyword(s) x "
                    f"{max(1, len(locations))} location(s) on {len(sites)} site(s)..."
                )
                summary = _run_with_override(
                    keywords=keywords,
                    locations=locations or None,
                    sites=sites or None,
                    enrich_limit=enrich_limit,
                    override=override if use_override else None,
                )
                st.write(
                    f"Scraped {summary.scraped_count} rows, "
                    f"enriched {summary.enriched_count} rows."
                )
                if summary.transform is not None:
                    st.write(
                        f"Transform: fact_rows={summary.transform.fact_rows}, "
                        f"bridge_rows={summary.transform.bridge_rows}, "
                        f"marts_refreshed={summary.transform.marts_refreshed}"
                    )
                status.update(label="Pipeline finished", state="complete")
                st.session_state[SESSION_KEY_LAST_SUMMARY] = summary.to_dict()
                st.success(
                    f"Run `{summary.run_id[:8]}` completed with status "
                    f"**{summary.status}**."
                )
                log.info(
                    "pipeline.ui.success",
                    run_id=summary.run_id,
                    scraped=summary.scraped_count,
                    enriched=summary.enriched_count,
                )
            except Exception as exc:
                status.update(label="Pipeline failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
                log.exception("pipeline.ui.failed")
            finally:
                st.session_state[SESSION_KEY_RUNNING] = False
                # Invalidate any cached data so downstream pages refresh.
                st.cache_data.clear()

    # ---- Last-run summary ------------------------------------------------ #
    last = st.session_state.get(SESSION_KEY_LAST_SUMMARY)
    if last:
        st.divider()
        st.subheader("Last run summary")
        st.json(last)


render()
