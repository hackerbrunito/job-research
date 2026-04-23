"""Search profiles page — CRUD for `user_search_profiles`.

Each profile bundles a named set of (keywords, locations, sites) the pipeline
will scrape against. The sidebar's active profile and the Run page both read
from the same table.
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from job_research import constants as C
from job_research.app.common import (
    Profile,
    create_profile_from_name,
    delete_profile,
    get_profile,
    list_profiles,
    save_profile,
)
from job_research.database import connect
from job_research.logging_setup import get_logger

log = get_logger(__name__)

_TEXTAREA_HEIGHT_PX: Final[int] = 200
_CREATE_OPTION: Final[str] = "__create_new__"
_SESSION_KEY_ACTIVE_PROFILE: Final[str] = "active_profile_id"


def _parse_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _options_with_create(profiles: list[Profile]) -> list[str]:
    return [p.profile_id for p in profiles] + [_CREATE_OPTION]


def _format_option(profiles: list[Profile], value: str) -> str:
    if value == _CREATE_OPTION:
        return "+ Create new..."
    for p in profiles:
        if p.profile_id == value:
            return p.name
    return value


def _render_create_form() -> None:
    st.subheader("Create a new profile")
    with st.form("create_profile_form", clear_on_submit=True):
        new_name = st.text_input(
            "Profile name",
            help="A short, human-readable name. The id is auto-generated.",
        )
        submitted = st.form_submit_button("Create profile")

    if not submitted:
        return
    if not new_name.strip():
        st.warning("Please provide a non-empty name.")
        return
    try:
        with connect(read_only=False) as con:
            profile = create_profile_from_name(con, new_name.strip())
        st.session_state[_SESSION_KEY_ACTIVE_PROFILE] = profile.profile_id
        st.cache_data.clear()
        st.success(f"Profile '{profile.name}' created.")
        log.info("profile.created", profile_id=profile.profile_id)
        st.rerun()
    except Exception as exc:
        log.error("profile.create.failed", error=str(exc))
        st.error(f"Failed to create profile: {exc}")


def _render_editor(profile: Profile) -> None:
    st.subheader(f"Edit: {profile.name}")

    with st.form("edit_profile_form", clear_on_submit=False):
        name = st.text_input("Name", value=profile.name)
        description = st.text_area(
            "Description",
            value=profile.description or "",
            height=80,
        )
        col_kw, col_loc = st.columns(2)
        with col_kw:
            keywords_raw = st.text_area(
                "Keywords (one per line)",
                value="\n".join(profile.keywords),
                height=_TEXTAREA_HEIGHT_PX,
            )
        with col_loc:
            locations_raw = st.text_area(
                "Locations (one per line)",
                value="\n".join(profile.locations),
                height=_TEXTAREA_HEIGHT_PX,
            )

        picked_sites = st.multiselect(
            "Sites",
            options=list(C.ALL_SITES),
            default=profile.sites or list(C.DEFAULT_SITES),
            help="Which job boards to scrape for this profile.",
        )

        submitted = st.form_submit_button("Save profile")

    keywords = _parse_lines(keywords_raw)
    locations = _parse_lines(locations_raw)
    sites_count = max(len(picked_sites), 1)
    n_requests = len(keywords) * max(len(locations), 1) * sites_count

    st.divider()
    st.subheader("Preview")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Keywords", len(keywords))
    col_b.metric("Locations", len(locations))
    col_c.metric("Sites", len(picked_sites))
    col_d.metric("Scrape requests", n_requests)
    st.caption(
        f"{len(keywords)} keywords x {max(len(locations), 1)} locations "
        f"x {sites_count} sites = {n_requests} scrape requests."
    )

    if submitted:
        if not name.strip():
            st.warning("Name cannot be blank.")
            return
        if not keywords:
            st.warning("Please provide at least one keyword.")
            return
        if not picked_sites:
            st.warning("Please select at least one site.")
            return
        updated = Profile(
            profile_id=profile.profile_id,
            name=name.strip(),
            description=description.strip() or None,
            keywords=keywords,
            locations=locations,
            sites=picked_sites,
        )
        try:
            with connect(read_only=False) as con:
                save_profile(con, updated)
            st.cache_data.clear()
            st.success("Profile saved.")
        except Exception as exc:
            log.error("profile.save.failed", error=str(exc))
            st.error(f"Failed to save profile: {exc}")

    st.divider()
    st.subheader("Danger zone")
    confirm = st.checkbox(
        f"I understand — delete profile '{profile.name}'.",
        key=f"confirm_delete_{profile.profile_id}",
    )
    if st.button("Delete profile", type="secondary", disabled=not confirm):
        try:
            with connect(read_only=False) as con:
                delete_profile(con, profile.profile_id)
            st.cache_data.clear()
            st.session_state.pop(_SESSION_KEY_ACTIVE_PROFILE, None)
            st.success(f"Profile '{profile.name}' deleted.")
            log.info("profile.deleted.ui", profile_id=profile.profile_id)
            st.rerun()
        except Exception as exc:
            log.error("profile.delete.failed", error=str(exc))
            st.error(f"Failed to delete profile: {exc}")


def main() -> None:
    st.title("Search profiles")
    st.write(
        "Manage saved search profiles. The active profile drives the sidebar "
        "status and seeds the *Run pipeline* page."
    )

    try:
        with connect(read_only=True) as con:
            profiles = list_profiles(con)
    except Exception as exc:
        log.error("profiles.load.failed", error=str(exc))
        st.error(f"Failed to load profiles: {exc}")
        return

    options = _options_with_create(profiles)
    active = st.session_state.get(_SESSION_KEY_ACTIVE_PROFILE)
    if active not in [p.profile_id for p in profiles]:
        active = options[0] if options else _CREATE_OPTION

    selection = st.selectbox(
        "Profile",
        options=options,
        index=options.index(active) if active in options else 0,
        format_func=lambda v: _format_option(profiles, v),
    )

    if selection == _CREATE_OPTION:
        _render_create_form()
        return

    # Load the freshest copy.
    try:
        with connect(read_only=True) as con:
            selected = get_profile(con, selection)
    except Exception as exc:
        log.error("profile.load.failed", error=str(exc))
        st.error(f"Failed to load profile: {exc}")
        return

    if selected is None:
        st.warning("That profile no longer exists.")
        return

    st.session_state[_SESSION_KEY_ACTIVE_PROFILE] = selected.profile_id
    _render_editor(selected)


main()
