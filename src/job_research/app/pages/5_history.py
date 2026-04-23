"""Run History page.

Read-only view of the `pipeline_runs` table: summary cards, filters, a
paginated table, and a run-detail expander.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Final

import pandas as pd
import streamlit as st

from job_research.app.common import list_profiles, read_only_connection
from job_research.database import connect
from job_research.logging_setup import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Page-local constants
# --------------------------------------------------------------------------- #
_CACHE_TTL_SECONDS: Final[int] = 30
_LIMIT_OPTIONS: Final[tuple[int, ...]] = (10, 25, 50, 100)
_DEFAULT_LIMIT: Final[int] = 50
_STATUS_VALUES: Final[tuple[str, ...]] = ("success", "failed", "running")
_RUN_ID_DISPLAY_CHARS: Final[int] = 8
_DEFAULT_DATE_WINDOW_DAYS: Final[int] = 30
_ALL_PROFILES_OPTION: Final[str] = "__all__"


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=_CACHE_TTL_SECONDS)
def _load_runs(limit: int) -> pd.DataFrame:
    """Fetch recent pipeline_runs rows as a DataFrame."""
    query = """
        SELECT
            run_id,
            profile_id,
            started_at,
            finished_at,
            status,
            keywords,
            locations,
            sites,
            scraped_count,
            enriched_count,
            error_message
        FROM pipeline_runs
        ORDER BY started_at DESC
        LIMIT ?
    """
    try:
        with connect(read_only=True) as con:
            df = con.execute(query, [limit]).fetchdf()
    except Exception as exc:
        log.debug("history.load.empty", error=str(exc))
        return pd.DataFrame(
            columns=[
                "run_id",
                "profile_id",
                "started_at",
                "finished_at",
                "status",
                "keywords",
                "locations",
                "sites",
                "scraped_count",
                "enriched_count",
                "error_message",
            ]
        )
    return df


@st.cache_data(ttl=_CACHE_TTL_SECONDS)
def _load_profile_name_map() -> dict[str, str]:
    """profile_id -> human-friendly name (empty if no profiles)."""
    try:
        with read_only_connection() as con:
            return {p.profile_id: p.name for p in list_profiles(con)}
    except Exception as exc:
        log.debug("history.profiles.load_failed", error=str(exc))
        return {}


def _coerce_list(cell: Any) -> list[str]:
    if cell is None:
        return []
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return [s]
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    return [str(cell)]


def _duration_seconds(started: Any, finished: Any) -> float | None:
    if started is None or finished is None:
        return None
    try:
        if isinstance(started, str):
            started = datetime.fromisoformat(started)
        if isinstance(finished, str):
            finished = datetime.fromisoformat(finished)
        if isinstance(started, pd.Timestamp):
            started = started.to_pydatetime()
        if isinstance(finished, pd.Timestamp):
            finished = finished.to_pydatetime()
        return (finished - started).total_seconds()
    except Exception:
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    return f"{minutes}m {sec}s"


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render() -> None:
    st.title("Run History")
    st.caption("Past pipeline executions recorded in `pipeline_runs`.")

    profile_name_map = _load_profile_name_map()

    # ---- Controls -------------------------------------------------------- #
    control_cols = st.columns([1, 1, 1, 1, 1])
    with control_cols[0]:
        limit = st.select_slider(
            "Rows",
            options=list(_LIMIT_OPTIONS),
            value=_DEFAULT_LIMIT,
        )
    with control_cols[1]:
        today = date.today()
        default_start = today - timedelta(days=_DEFAULT_DATE_WINDOW_DAYS)
        date_range = st.date_input(
            "Date range",
            value=(default_start, today),
            help="Filter runs by started_at date.",
        )
    with control_cols[2]:
        statuses = st.multiselect(
            "Status",
            options=list(_STATUS_VALUES),
            default=list(_STATUS_VALUES),
        )
    with control_cols[3]:
        profile_options = [_ALL_PROFILES_OPTION, *profile_name_map.keys()]
        profile_labels = {_ALL_PROFILES_OPTION: "All profiles", **profile_name_map}
        profile_choice = st.selectbox(
            "Profile",
            options=profile_options,
            index=0,
            format_func=lambda v: profile_labels.get(v, v),
        )
    with control_cols[4]:
        if st.button("Refresh", width="stretch"):
            _load_runs.clear()
            _load_profile_name_map.clear()
            st.rerun()

    df = _load_runs(int(limit))

    if df.empty:
        st.info(
            "No pipeline runs recorded yet. Trigger your first run from the "
            "**Run Pipeline** page."
        )
        return

    # ---- Derive helper columns ------------------------------------------- #
    df = df.copy()
    df["keywords_list"] = df["keywords"].apply(_coerce_list)
    df["locations_list"] = df["locations"].apply(_coerce_list)
    df["sites_list"] = df["sites"].apply(_coerce_list)
    df["duration_seconds"] = [
        _duration_seconds(s, f)
        for s, f in zip(df["started_at"], df["finished_at"], strict=False)
    ]
    df["duration"] = df["duration_seconds"].apply(_format_duration)
    df["run_id_short"] = df["run_id"].str[:_RUN_ID_DISPLAY_CHARS]
    df["keywords_str"] = df["keywords_list"].apply(lambda xs: ", ".join(xs))
    df["profile_name"] = df["profile_id"].apply(
        lambda pid: profile_name_map.get(pid, pid) if pid else "-"
    )

    # ---- Apply filters --------------------------------------------------- #
    if statuses:
        df = df[df["status"].isin(statuses)]

    if profile_choice != _ALL_PROFILES_OPTION:
        df = df[df["profile_id"] == profile_choice]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        started_dt = pd.to_datetime(df["started_at"], errors="coerce")
        mask = (started_dt.dt.date >= start_date) & (started_dt.dt.date <= end_date)
        df = df[mask]

    # ---- Summary cards --------------------------------------------------- #
    total = len(df)
    success_count = int((df["status"] == "success").sum())
    success_rate = (success_count / total * 100) if total else 0.0
    avg_duration = df["duration_seconds"].dropna().mean() if not df.empty else None
    total_scraped = int(df["scraped_count"].fillna(0).sum())

    card_cols = st.columns(4)
    card_cols[0].metric("Total runs", total)
    card_cols[1].metric("Success rate", f"{success_rate:.0f}%")
    card_cols[2].metric(
        "Avg duration",
        _format_duration(float(avg_duration)) if avg_duration is not None else "-",
    )
    card_cols[3].metric("Total scraped", total_scraped)

    st.divider()

    # ---- Table ----------------------------------------------------------- #
    st.subheader("Runs")
    display_cols = [
        "run_id_short",
        "profile_name",
        "started_at",
        "duration",
        "status",
        "keywords_str",
        "scraped_count",
        "enriched_count",
        "error_message",
    ]
    renamed = df[display_cols].rename(
        columns={
            "run_id_short": "run_id",
            "profile_name": "profile",
            "started_at": "started_at",
            "duration": "duration",
            "status": "status",
            "keywords_str": "keywords",
            "scraped_count": "scraped",
            "enriched_count": "enriched",
            "error_message": "error",
        }
    )
    st.dataframe(renamed, width="stretch", hide_index=True)

    # ---- Detail expander ------------------------------------------------- #
    st.subheader("Run detail")
    if df.empty:
        st.caption("No rows match the current filters.")
        return

    options = df["run_id"].tolist()
    labels = {
        r: f"{r[:_RUN_ID_DISPLAY_CHARS]}  ({s})"
        for r, s in zip(df["run_id"], df["status"], strict=False)
    }
    selected = st.selectbox(
        "Select a run",
        options=options,
        format_func=lambda r: labels.get(r, r),
    )

    row = df[df["run_id"] == selected].iloc[0]
    st.markdown(f"**run_id:** `{row['run_id']}`")
    st.markdown(f"**profile:** {row['profile_name']}")
    st.markdown(f"**started_at:** {row['started_at']}")
    st.markdown(f"**finished_at:** {row['finished_at']}")
    st.markdown(f"**duration:** {row['duration']}")
    st.markdown(f"**status:** {row['status']}")
    st.markdown(f"**scraped:** {row['scraped_count']}")
    st.markdown(f"**enriched:** {row['enriched_count']}")

    col_kw, col_loc, col_sites = st.columns(3)
    with col_kw:
        st.markdown("**Keywords**")
        st.write(row["keywords_list"] or "-")
    with col_loc:
        st.markdown("**Locations**")
        st.write(row["locations_list"] or "-")
    with col_sites:
        st.markdown("**Sites**")
        st.write(row["sites_list"] or "-")

    if row.get("error_message"):
        st.markdown("**Error message**")
        st.code(str(row["error_message"]))


render()
