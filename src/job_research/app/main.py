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
    cached_count_jobs,
    cached_latest_run_status,
)
from job_research.config import get_settings
from job_research.database import init_schema
from job_research.logging_setup import configure_logging, get_logger

log = get_logger(__name__)

_PAGES_DIR: Final[Path] = Path(__file__).parent / "pages"
_SESSION_KEY_BOOTSTRAPPED: Final[str] = "_job_research_bootstrapped"


def _bootstrap_once() -> None:
    """Run one-time init per Streamlit session: logging + schema."""
    if st.session_state.get(_SESSION_KEY_BOOTSTRAPPED):
        return

    settings = get_settings()
    configure_logging(level=settings.log_level)
    init_schema()
    st.session_state[_SESSION_KEY_BOOTSTRAPPED] = True
    log.info("app.bootstrap.done", db_path=str(settings.database.path))


def _render_sidebar_status() -> None:
    """Show application status (LLM, DB, job count, last run) in the sidebar."""
    settings = get_settings()

    with st.sidebar:
        st.subheader("Status")

        st.caption("LLM")
        st.write(f"Provider: `{settings.llm.provider}`")
        st.write(f"Model: `{settings.llm.model}`")

        st.caption("Database")
        st.write(f"Path: `{settings.database.path}`")

        try:
            jobs_count = cached_count_jobs()
        except Exception as exc:
            log.warning("sidebar.count_jobs.failed", error=str(exc))
            jobs_count = 0
        st.metric("Jobs in database", jobs_count)

        st.caption("Last pipeline run")
        try:
            last_run = cached_latest_run_status()
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
        ],
    }

    # Drop any sections whose page files don't exist yet (Agent B may not have
    # landed their pages yet during parallel development).
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
                ][: len(section_pages)],
                strict=False,
            )
            if src.is_file()
        ]
        # Fallback — if no filtering matched (shouldn't happen), keep originals.
        if existing:
            filtered[section] = existing

    pg = st.navigation(filtered or pages, position="sidebar")
    pg.run()


main()
