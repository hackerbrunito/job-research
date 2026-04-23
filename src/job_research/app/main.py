"""Streamlit entrypoint — multi-page app router.

Run with:
    streamlit run src/job_research/app/main.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import streamlit as st

from job_research import constants as C
from job_research.app.common import (
    Profile,
    cached_count_jobs,
    cached_latest_run_status,
    ensure_default_profile,
    list_profiles,
)
from job_research.config import get_settings
from job_research.database import connect, init_schema
from job_research.logging_setup import configure_logging, get_logger

log = get_logger(__name__)

_PAGES_DIR: Final[Path] = Path(__file__).parent / "pages"
_SESSION_KEY_BOOTSTRAPPED: Final[str] = "_job_research_bootstrapped"
_SESSION_KEY_ACTIVE_PROFILE: Final[str] = "active_profile_id"


def _bootstrap_once() -> None:
    """Run one-time init per Streamlit session: logging + schema + default profile."""
    if st.session_state.get(_SESSION_KEY_BOOTSTRAPPED):
        return

    settings = get_settings()
    configure_logging(level=settings.log_level)
    init_schema()
    # Seed a default profile so pages can rely on at least one existing.
    try:
        with connect(read_only=False) as con:
            ensure_default_profile(con)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("app.bootstrap.default_profile_failed", error=str(exc))
    st.session_state[_SESSION_KEY_BOOTSTRAPPED] = True
    log.info("app.bootstrap.done", db_path=str(settings.database.path))


def _load_profiles_safely() -> list[Profile]:
    try:
        from job_research.app.common import read_only_connection

        with read_only_connection() as con:
            return list_profiles(con)
    except Exception as exc:
        log.warning("sidebar.list_profiles.failed", error=str(exc))
        return []


def _render_sidebar_status() -> None:
    """Show application status (profile, LLM, DB, job count, last run) in the sidebar."""
    settings = get_settings()

    with st.sidebar:
        st.subheader("Profile")

        profiles = _load_profiles_safely()
        if profiles:
            ids = [p.profile_id for p in profiles]
            labels = {p.profile_id: p.name for p in profiles}

            active = st.session_state.get(_SESSION_KEY_ACTIVE_PROFILE)
            if active not in ids:
                active = ids[0]

            picked = st.selectbox(
                "Current profile",
                options=ids,
                index=ids.index(active),
                format_func=lambda pid: labels.get(pid, pid),
                key="sidebar_profile_select",
            )
            st.session_state[_SESSION_KEY_ACTIVE_PROFILE] = picked
            active_profile_id: str | None = picked
        else:
            st.caption("No profiles yet. Create one on the Search config page.")
            active_profile_id = None

        st.divider()
        st.subheader("Status")

        st.caption("LLM")
        st.write(f"Provider: `{settings.llm.provider}`")
        st.write(f"Model: `{settings.llm.model}`")

        st.caption("Database")
        st.write(f"Path: `{settings.database.path}`")

        try:
            jobs_count = cached_count_jobs(profile_id=active_profile_id)
        except Exception as exc:
            log.warning("sidebar.count_jobs.failed", error=str(exc))
            jobs_count = 0
        st.metric("Jobs in database", jobs_count)

        st.caption("Last pipeline run")
        try:
            last_run = cached_latest_run_status(profile_id=active_profile_id)
        except Exception as exc:
            log.warning("sidebar.last_run.failed", error=str(exc))
            last_run = None

        if last_run is None:
            st.write("No runs yet")
        else:
            st.write(f"Status: `{last_run['status']}`")
            if last_run["started_at"]:
                st.write(f"Started: {last_run['started_at']}")

        st.divider()
        st.caption(f"{C.APP_NAME} v{C.APP_VERSION}")


def main() -> None:
    st.set_page_config(
        page_title=C.APP_NAME,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _bootstrap_once()
    _render_sidebar_status()

    pages = {
        "Analyze": [
            st.Page(
                str(_PAGES_DIR / "1_results.py"),
                title="Results",
                url_path="results",
                default=True,
            ),
        ],
        "Configure": [
            st.Page(
                str(_PAGES_DIR / "2_search.py"),
                title="Search config",
                url_path="search",
            ),
            st.Page(
                str(_PAGES_DIR / "3_llm.py"),
                title="LLM settings",
                url_path="llm",
            ),
        ],
        "Operate": [
            st.Page(
                str(_PAGES_DIR / "4_run.py"),
                title="Run pipeline",
                url_path="run",
            ),
            st.Page(
                str(_PAGES_DIR / "5_history.py"),
                title="Run history",
                url_path="history",
            ),
            st.Page(
                str(_PAGES_DIR / "6_triage.py"),
                title="Triage",
                url_path="triage",
            ),
        ],
    }

    # Drop any sections whose page files don't exist yet.
    filtered: dict[str, list[st.Page]] = {}
    for section, section_pages in pages.items():
        existing = [
            p
            for p, src in zip(
                section_pages,
                [
                    _PAGES_DIR / "1_results.py",
                    _PAGES_DIR / "2_search.py",
                    _PAGES_DIR / "3_llm.py",
                    _PAGES_DIR / "4_run.py",
                    _PAGES_DIR / "5_history.py",
                    _PAGES_DIR / "6_triage.py",
                ][: len(section_pages)],
                strict=False,
            )
            if src.is_file()
        ]
        if existing:
            filtered[section] = existing

    pg = st.navigation(filtered or pages, position="sidebar")
    pg.run()


main()
