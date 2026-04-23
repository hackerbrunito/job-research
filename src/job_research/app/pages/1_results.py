"""Results dashboard — charts + data tables across all scraped jobs."""

from __future__ import annotations

from typing import Final

import pandas as pd
import plotly.express as px
import streamlit as st

from job_research import constants as C
from job_research.app.common import (
    DEFAULT_FACT_LIMIT,
    MAX_FACT_LIMIT,
    cached_fact_with_dims,
    cached_mart,
)
from job_research.logging_setup import get_logger

log = get_logger(__name__)

# ---- Page-local constants ------------------------------------------------
TOP_TECH_SKILLS: Final[int] = 20
TOP_SOFT_SKILLS: Final[int] = 10
TABLE_HEIGHT_PX: Final[int] = 520


def _show_empty_state() -> None:
    st.info(
        "No jobs yet. Configure your search on the *Search config* page, "
        "then trigger a run from the *Run pipeline* page."
    )


def _render_filters(fact_df: pd.DataFrame) -> dict[str, object]:
    """Top-of-page filter widgets. Returns the selections as a dict."""
    available_keywords = (
        sorted(fact_df["search_keyword"].dropna().unique().tolist())
        if not fact_df.empty
        else []
    )
    available_countries = (
        sorted(fact_df["country"].dropna().unique().tolist())
        if not fact_df.empty
        else []
    )
    work_modes = sorted(C.WORK_MODES)

    col_kw, col_country, col_mode = st.columns([2, 2, 1])
    with col_kw:
        picked_keywords = st.multiselect(
            "Search keyword",
            options=available_keywords,
            default=available_keywords,
        )
    with col_country:
        picked_countries = st.multiselect(
            "Country",
            options=available_countries,
            default=[],
        )
    with col_mode:
        picked_modes = st.multiselect("Work mode", options=work_modes, default=[])

    return {
        "keywords": picked_keywords,
        "countries": picked_countries,
        "work_modes": picked_modes,
    }


def _apply_filters(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    keywords = filters.get("keywords") or []
    countries = filters.get("countries") or []
    modes = filters.get("work_modes") or []
    if keywords:
        out = out[out["search_keyword"].isin(keywords)]
    if countries and "country" in out.columns:
        out = out[out["country"].isin(countries)]
    if modes and "work_mode" in out.columns:
        out = out[out["work_mode"].isin(modes)]
    return out


def _filter_mart_by_keyword(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if df.empty or not keywords:
        return df
    if "search_keyword" not in df.columns:
        return df
    return df[df["search_keyword"].isin(keywords)]


def _render_country_map(jobs_by_country: pd.DataFrame) -> None:
    st.subheader("Jobs by country")
    if jobs_by_country.empty:
        st.caption("No country data yet.")
        return
    agg = (
        jobs_by_country.groupby(["country_code", "country"], as_index=False)[
            "job_count"
        ]
        .sum()
        .sort_values("job_count", ascending=False)
    )
    # Marts store ISO 3166-1 alpha-2; Plotly ISO-3 choropleth needs alpha-3.
    # Convert with pycountry (already a dep); rows that don't resolve are dropped.
    import pycountry

    def _alpha3(code: str | None) -> str | None:
        if not code:
            return None
        try:
            return pycountry.countries.get(alpha_2=code.upper()).alpha_3  # type: ignore[union-attr]
        except (AttributeError, KeyError):
            return None

    agg = agg.assign(_iso3=agg["country_code"].map(_alpha3)).dropna(subset=["_iso3"])
    if agg.empty:
        st.caption("No country codes resolved to ISO alpha-3 yet.")
        return
    fig = px.choropleth(
        agg,
        locations="_iso3",
        locationmode="ISO-3",
        color="job_count",
        hover_name="country",
        color_continuous_scale="Blues",
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def _render_skills(
    skills_df: pd.DataFrame, *, skill_type: str, top_n: int, title: str
) -> None:
    st.subheader(title)
    if skills_df.empty:
        st.caption("No skill data yet.")
        return
    subset = skills_df[skills_df["skill_type"] == skill_type]
    if subset.empty:
        st.caption(f"No {skill_type} skill data yet.")
        return

    keywords = sorted(subset["search_keyword"].dropna().unique().tolist())
    if not keywords:
        return

    picked = st.selectbox(
        f"Keyword ({skill_type} skills)",
        options=keywords,
        key=f"skills_kw_{skill_type}",
    )
    per_kw = (
        subset[subset["search_keyword"] == picked]
        .nlargest(top_n, "demand_count")
        .sort_values("demand_count")
    )
    fig = px.bar(
        per_kw,
        x="demand_count",
        y="skill_name",
        orientation="h",
        labels={"demand_count": "Demand", "skill_name": "Skill"},
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def _render_salary(salary_df: pd.DataFrame) -> None:
    st.subheader("Salary ranges (p25 - p75)")
    if salary_df.empty:
        st.caption("No salary data yet.")
        return
    # Build a long-form frame: for each keyword+currency, two bands: min & max.
    rows: list[dict[str, object]] = []
    for _, r in salary_df.iterrows():
        label = f"{r['search_keyword']} ({r['currency']}/{r['period']})"
        rows.append(
            {
                "label": label,
                "band": "min",
                "p25": r["p25_min"],
                "p50": r["p50_min"],
                "p75": r["p75_min"],
            }
        )
        rows.append(
            {
                "label": label,
                "band": "max",
                "p25": r["p25_max"],
                "p50": r["p50_max"],
                "p75": r["p75_max"],
            }
        )
    long_df = pd.DataFrame(rows).dropna(subset=["p25", "p75"])
    if long_df.empty:
        st.caption("No salary percentiles available.")
        return

    fig = px.bar(
        long_df,
        x="label",
        y="p75",
        color="band",
        barmode="group",
        hover_data=["p25", "p50", "p75"],
        labels={"p75": "p75", "label": "Keyword / currency"},
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def _render_work_mode(work_mode_df: pd.DataFrame) -> None:
    st.subheader("Work mode mix")
    if work_mode_df.empty:
        st.caption("No work mode data yet.")
        return
    fig = px.bar(
        work_mode_df,
        x="search_keyword",
        y="job_count",
        color="work_mode",
        barmode="stack",
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def _render_table(fact_df: pd.DataFrame) -> None:
    st.subheader("Job listings")
    if fact_df.empty:
        st.caption("No jobs match the current filters.")
        return

    display = fact_df.copy()
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=TABLE_HEIGHT_PX,
        column_config={
            "job_url": st.column_config.LinkColumn("Job URL"),
            "date_posted": st.column_config.DateColumn("Date posted"),
            "scraped_at": st.column_config.DatetimeColumn("Scraped at"),
            "enriched_at": st.column_config.DatetimeColumn("Enriched at"),
            "salary_min": st.column_config.NumberColumn("Salary min"),
            "salary_max": st.column_config.NumberColumn("Salary max"),
        },
    )


def main() -> None:
    st.title("Results")

    # Load data
    try:
        fact_df = cached_fact_with_dims(keyword=None, limit=DEFAULT_FACT_LIMIT)
        jobs_by_country = cached_mart("mart_jobs_by_country")
        skills_df = cached_mart("mart_skills_by_keyword")
        salary_df = cached_mart("mart_salary_by_keyword")
        work_mode_df = cached_mart("mart_work_mode_distribution")
    except Exception as exc:
        log.error("results.load.failed", error=str(exc))
        st.error(f"Failed to load data: {exc}")
        return

    if fact_df.empty:
        _show_empty_state()
        return

    filters = _render_filters(fact_df)
    filtered_fact = _apply_filters(fact_df, filters)
    selected_keywords = filters.get("keywords") or []
    assert isinstance(selected_keywords, list)

    filtered_country = _filter_mart_by_keyword(jobs_by_country, selected_keywords)
    filtered_skills = _filter_mart_by_keyword(skills_df, selected_keywords)
    filtered_salary = _filter_mart_by_keyword(salary_df, selected_keywords)
    filtered_work_mode = _filter_mart_by_keyword(work_mode_df, selected_keywords)

    tab_map, tab_skills, tab_salary, tab_mode, tab_table = st.tabs(
        ["Map", "Skills", "Salary", "Work mode", "Table"]
    )

    with tab_map:
        _render_country_map(filtered_country)
    with tab_skills:
        col_tech, col_soft = st.columns(2)
        with col_tech:
            _render_skills(
                filtered_skills,
                skill_type=C.SKILL_TYPE_TECH,
                top_n=TOP_TECH_SKILLS,
                title=f"Top {TOP_TECH_SKILLS} tech skills",
            )
        with col_soft:
            _render_skills(
                filtered_skills,
                skill_type=C.SKILL_TYPE_SOFT,
                top_n=TOP_SOFT_SKILLS,
                title=f"Top {TOP_SOFT_SKILLS} soft skills",
            )
    with tab_salary:
        _render_salary(filtered_salary)
    with tab_mode:
        _render_work_mode(filtered_work_mode)
    with tab_table:
        limit = st.slider(
            "Row limit",
            min_value=50,
            max_value=MAX_FACT_LIMIT,
            value=DEFAULT_FACT_LIMIT,
            step=50,
        )
        if limit != DEFAULT_FACT_LIMIT:
            try:
                fact_df = cached_fact_with_dims(keyword=None, limit=limit)
                filtered_fact = _apply_filters(fact_df, filters)
            except Exception as exc:
                log.error("results.reload.failed", error=str(exc))
                st.error(f"Failed to reload table: {exc}")
        _render_table(filtered_fact)


main()
