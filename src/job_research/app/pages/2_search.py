"""Search configuration page — edit keywords, locations, and sites."""

from __future__ import annotations

from typing import Final

import streamlit as st

from job_research import constants as C
from job_research.app.common import (
    get_search_config,
    read_only_connection,
    save_search_config,
)
from job_research.database import connect
from job_research.logging_setup import get_logger

log = get_logger(__name__)

_TEXTAREA_HEIGHT_PX: Final[int] = 200


def _parse_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def main() -> None:
    st.title("Search configuration")
    st.write(
        "Define what the pipeline should scrape. These settings are read by "
        "the *Run pipeline* page."
    )

    try:
        with read_only_connection() as con:
            current = get_search_config(con)
    except Exception as exc:
        log.error("search_config.load.failed", error=str(exc))
        st.error(f"Failed to load current config: {exc}")
        return

    with st.form("search_config_form", clear_on_submit=False):
        col_kw, col_loc = st.columns(2)
        with col_kw:
            keywords_raw = st.text_area(
                "Keywords (one per line)",
                value="\n".join(current["keywords"]),
                height=_TEXTAREA_HEIGHT_PX,
            )
        with col_loc:
            locations_raw = st.text_area(
                "Locations (one per line)",
                value="\n".join(current["locations"]),
                height=_TEXTAREA_HEIGHT_PX,
            )

        picked_sites = st.multiselect(
            "Sites",
            options=list(C.ALL_SITES),
            default=current["sites"] or list(C.DEFAULT_SITES),
            help="Select which job boards to scrape.",
        )

        submitted = st.form_submit_button("Save configuration")

    keywords = _parse_lines(keywords_raw)
    locations = _parse_lines(locations_raw)
    n_requests = len(keywords) * len(locations) * max(len(picked_sites), 1)

    st.divider()
    st.subheader("Preview")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Keywords", len(keywords))
    col_b.metric("Locations", len(locations))
    col_c.metric("Sites", len(picked_sites))
    col_d.metric("Scrape requests", n_requests)
    st.caption(
        "A scrape request is one (keyword, location, site) combination. "
        "JobSpy is called once per request."
    )

    if submitted:
        if not keywords:
            st.warning("Please provide at least one keyword.")
            return
        if not locations:
            st.warning("Please provide at least one location.")
            return
        if not picked_sites:
            st.warning("Please select at least one site.")
            return

        try:
            with connect(read_only=False) as con:
                save_search_config(
                    con,
                    keywords=keywords,
                    locations=locations,
                    sites=picked_sites,
                )
            st.cache_data.clear()
            st.success("Configuration saved.")
        except Exception as exc:
            log.error("search_config.save.failed", error=str(exc))
            st.error(f"Failed to save configuration: {exc}")


main()
