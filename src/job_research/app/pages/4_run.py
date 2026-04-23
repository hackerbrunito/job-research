"""Run Pipeline page.

Trigger the scrape -> enrich -> transform pipeline against one or more
search profiles. Each selected profile becomes its own pipeline run.

Design rules:
- Pipeline runs ONLY on button click, never on page load.
- The run button is disabled while a run is in progress.
- After all runs finish, `st.cache_data` is cleared so Results and History
  reflect the new state.
"""

from __future__ import annotations

from typing import Any, Final

import streamlit as st

from job_research.app.common import Profile, list_profiles, read_only_connection
from job_research.config import LLMConfig, Settings
from job_research.logging_setup import get_logger
from job_research.pipeline import PipelineSummary, job_research_pipeline

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Page-local constants
# --------------------------------------------------------------------------- #
SESSION_KEY_RUNNING: Final[str] = "pipeline_running"
SESSION_KEY_LAST_SUMMARIES: Final[str] = "pipeline_last_summaries"
SESSION_KEY_LLM_OVERRIDE: Final[str] = "llm_override"
SESSION_KEY_ACTIVE_PROFILE: Final[str] = "active_profile_id"

_ENRICH_LIMIT_MIN: Final[int] = 1
_ENRICH_LIMIT_MAX: Final[int] = 10_000
_ENRICH_LIMIT_DEFAULT: Final[int] = 50
_RUN_ID_DISPLAY_CHARS: Final[int] = 8


# --------------------------------------------------------------------------- #
# Profile retrieval
# --------------------------------------------------------------------------- #
def _load_profiles() -> list[Profile]:
    try:
        with read_only_connection() as con:
            return list_profiles(con)
    except Exception as exc:
        log.debug("run.profiles.unavailable", error=str(exc))
        return []


# --------------------------------------------------------------------------- #
# LLM override application
# --------------------------------------------------------------------------- #
def _apply_llm_override(settings: Settings, override: dict[str, Any]) -> Settings:
    """Return a Settings copy with LLMConfig merged from the override."""
    merged = settings.llm.model_dump()
    merged.update({k: v for k, v in override.items() if v is not None})
    new_llm = LLMConfig(**merged)
    return settings.model_copy(update={"llm": new_llm})


def _run_one(
    *,
    profile: Profile,
    enrich_limit: int | None,
    override: dict[str, Any] | None,
) -> PipelineSummary:
    """Run the pipeline once for a single profile."""
    from job_research.config import get_settings

    base = get_settings()
    effective = _apply_llm_override(base, override) if override else base
    return job_research_pipeline(
        keywords=profile.keywords,
        locations=profile.locations or None,
        sites=profile.sites or None,
        enrich_limit=enrich_limit,
        settings=effective,
        profile_id=profile.profile_id,
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render() -> None:
    st.title("Run Pipeline")
    st.caption("Trigger the scrape -> enrich -> transform pipeline on demand.")

    profiles = _load_profiles()
    if not profiles:
        st.warning("No profiles found. Create one on the **Search config** page first.")
        return

    # ---- Profile selection ---------------------------------------------- #
    st.subheader("Profiles to run")
    ids = [p.profile_id for p in profiles]
    by_id: dict[str, Profile] = {p.profile_id: p for p in profiles}

    active = st.session_state.get(SESSION_KEY_ACTIVE_PROFILE)
    default_selection = [active] if active in ids else [ids[0]]

    picked_ids = st.multiselect(
        "Profiles",
        options=ids,
        default=default_selection,
        format_func=lambda pid: by_id[pid].name,
        help="Each selected profile becomes its own pipeline run.",
    )

    if not picked_ids:
        st.info("Select at least one profile to enable the Run button.")
        return

    # ---- Planned workload preview --------------------------------------- #
    total_requests = 0
    for pid in picked_ids:
        p = by_id[pid]
        total_requests += len(p.keywords) * max(len(p.locations), 1)

    st.info(
        f"Planned: {len(picked_ids)} run(s), "
        f"{total_requests} scrape request(s) in total."
    )

    with st.expander("Profile details", expanded=False):
        for pid in picked_ids:
            p = by_id[pid]
            st.markdown(f"**{p.name}** (`{p.profile_id}`)")
            st.write(
                f"- keywords: {', '.join(p.keywords) or '(none)'}\n"
                f"- locations: {', '.join(p.locations) or '(none - global)'}\n"
                f"- sites: {', '.join(p.sites)}"
            )

    st.divider()

    # ---- Run options ---------------------------------------------------- #
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
                "(provider, model, etc.) for these runs only."
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

    # ---- Trigger -------------------------------------------------------- #
    running = bool(st.session_state.get(SESSION_KEY_RUNNING, False))
    run_clicked = st.button(
        "Run pipeline",
        type="primary",
        disabled=running,
        width="content",
    )

    if run_clicked:
        st.session_state[SESSION_KEY_RUNNING] = True
        summaries: list[dict[str, Any]] = []
        container = st.container()
        try:
            for pid in picked_ids:
                profile = by_id[pid]
                with container.status(
                    f"Running pipeline for '{profile.name}'...",
                    expanded=True,
                ) as status:
                    try:
                        st.write(
                            f"Scraping {len(profile.keywords)} keyword(s) x "
                            f"{max(1, len(profile.locations))} location(s) "
                            f"on {len(profile.sites)} site(s)..."
                        )
                        summary = _run_one(
                            profile=profile,
                            enrich_limit=enrich_limit,
                            override=override if use_override else None,
                        )
                        st.write(
                            f"Scraped {summary.scraped_count} rows, "
                            f"enriched {summary.enriched_count} rows."
                        )
                        if summary.transform is not None:
                            st.write(
                                "Transform: "
                                f"fact_rows={summary.transform.fact_rows}, "
                                f"bridge_rows={summary.transform.bridge_rows}, "
                                "marts_refreshed="
                                f"{summary.transform.marts_refreshed}"
                            )
                        status.update(
                            label=f"'{profile.name}' finished",
                            state="complete",
                        )
                        summaries.append(
                            {
                                "profile_id": profile.profile_id,
                                "profile_name": profile.name,
                                **summary.to_dict(),
                            }
                        )
                        log.info(
                            "pipeline.ui.success",
                            run_id=summary.run_id,
                            profile_id=profile.profile_id,
                            scraped=summary.scraped_count,
                            enriched=summary.enriched_count,
                        )
                    except Exception as exc:
                        status.update(
                            label=f"'{profile.name}' failed",
                            state="error",
                        )
                        st.error(f"{type(exc).__name__}: {exc}")
                        log.exception(
                            "pipeline.ui.failed", profile_id=profile.profile_id
                        )
                        summaries.append(
                            {
                                "profile_id": profile.profile_id,
                                "profile_name": profile.name,
                                "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
        finally:
            st.session_state[SESSION_KEY_RUNNING] = False
            st.session_state[SESSION_KEY_LAST_SUMMARIES] = summaries
            st.cache_data.clear()

        ok = sum(1 for s in summaries if s.get("status") == "success")
        st.success(f"Completed {ok}/{len(summaries)} run(s) successfully.")

    # ---- Last-run summaries -------------------------------------------- #
    last = st.session_state.get(SESSION_KEY_LAST_SUMMARIES)
    if last:
        st.divider()
        st.subheader("Last run summaries")
        st.json(last)


render()
