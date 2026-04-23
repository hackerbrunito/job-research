"""SQL transformation layer — staging + intermediate → dims + facts + marts.

End-to-end idempotent transform. Callable from Prefect or tests via
`run_transform()`. Surrogate keys are computed in Python using
`database.stable_key()` (blake2b, 12 bytes) because blake2b has no pure
SQL equivalent — we generate them once, then hand DataFrames to the
DML files which only do `INSERT ... ON CONFLICT DO NOTHING`.

Mart refreshes are pure SQL (DELETE + INSERT full refresh — cheap at our
scale and simpler than MERGE).
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import duckdb
import pandas as pd

from job_research.constants import SKILL_TYPE_SOFT, SKILL_TYPE_TECH
from job_research.database import connect, load_sql, stable_key
from job_research.logging_setup import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass
class TransformSummary:
    """Row counts produced by a transform pass. Used by the pipeline flow."""

    dim_location_rows: int
    dim_salary_rows: int
    dim_skill_rows: int
    fact_rows: int
    bridge_rows: int
    marts_refreshed: int


_MART_FILES: tuple[str, ...] = (
    "30_refresh_mart_jobs_by_country",
    "31_refresh_mart_skills_by_keyword",
    "32_refresh_mart_salary_by_keyword",
    "33_refresh_mart_work_mode_distribution",
)


def run_transform(
    *,
    con: duckdb.DuckDBPyConnection | None = None,
) -> TransformSummary:
    """Run the full transform: upsert dims, upsert fact + bridge, refresh marts.

    Idempotent — safe to re-run. Uses the provided connection, or opens one
    via `database.connect()` if `con` is None.
    """
    cm = nullcontext(con) if con is not None else connect()
    with cm as c:
        assert c is not None  # narrow for type checker
        return _run(c)


# --------------------------------------------------------------------------- #
# Internal
# --------------------------------------------------------------------------- #
def _run(con: duckdb.DuckDBPyConnection) -> TransformSummary:
    log.info("transform.start")

    enriched = _load_enriched(con)
    if enriched.empty:
        log.info("transform.no_enriched_rows")
        # Still refresh marts (they may need to be emptied).
        marts = _refresh_marts(con)
        return TransformSummary(0, 0, 0, 0, 0, marts)

    dim_loc = _build_dim_location(enriched)
    dim_sal = _build_dim_salary(enriched)
    dim_skill, bridge = _build_dim_skill_and_bridge(enriched)
    fact = _build_fact(enriched, dim_loc, dim_sal)

    dim_loc_inserted = _upsert(
        con, dim_loc, "_dim_location_df", "10_upsert_dim_location"
    )
    dim_sal_inserted = _upsert(con, dim_sal, "_dim_salary_df", "11_upsert_dim_salary")
    dim_skill_inserted = _upsert(con, dim_skill, "_dim_skill_df", "12_upsert_dim_skill")
    fact_inserted = _upsert(con, fact, "_fact_df", "20_upsert_fact")
    bridge_inserted = _upsert(con, bridge, "_bridge_df", "21_upsert_bridge")

    marts = _refresh_marts(con)

    summary = TransformSummary(
        dim_location_rows=dim_loc_inserted,
        dim_salary_rows=dim_sal_inserted,
        dim_skill_rows=dim_skill_inserted,
        fact_rows=fact_inserted,
        bridge_rows=bridge_inserted,
        marts_refreshed=marts,
    )
    log.info("transform.done", **summary.__dict__)
    return summary


def _load_enriched(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load the staging+intermediate join into a DataFrame."""
    return con.execute(
        """
        SELECT
            s.id              AS job_id,
            s.run_id,
            s.scraped_at,
            s.site,
            s.search_keyword,
            s.company,
            s.title,
            s.job_url,
            s.date_posted,
            i.enriched_at,
            i.tech_skills,
            i.soft_skills,
            i.city,
            i.country,
            i.country_code,
            i.work_mode,
            i.salary_min,
            i.salary_max,
            i.salary_currency,
            i.salary_period
        FROM staging_job_offers s
        JOIN int_enriched_job_info i ON s.id = i.job_id
        """
    ).df()


# ---- dim builders --------------------------------------------------------- #
def _norm(val: Any) -> str:
    """Lowercased trimmed string, empty if null/blank."""
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    s = str(val).strip().lower()
    return s


def _build_dim_location(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for _, r in df.iterrows():
        city_n = _norm(r["city"])
        cc_n = _norm(r["country_code"])
        key = stable_key(city_n, cc_n)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "location_key": key,
                "city": r["city"] if city_n else None,
                "country": r["country"] if _norm(r["country"]) else None,
                "country_code": r["country_code"] if cc_n else None,
            }
        )
    return pd.DataFrame(
        rows, columns=["location_key", "city", "country", "country_code"]
    )


def _build_dim_salary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for _, r in df.iterrows():
        smin = r["salary_min"]
        smax = r["salary_max"]
        cur_n = _norm(r["salary_currency"])
        per_n = _norm(r["salary_period"])
        smin_norm = None if pd.isna(smin) else float(smin)
        smax_norm = None if pd.isna(smax) else float(smax)
        key = stable_key(smin_norm, smax_norm, cur_n, per_n)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "salary_key": key,
                "min_amount": smin_norm,
                "max_amount": smax_norm,
                "currency": r["salary_currency"] if cur_n else None,
                "period": r["salary_period"] if per_n else None,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["salary_key", "min_amount", "max_amount", "currency", "period"],
    )


def _parse_skills(raw: Any) -> list[str]:
    """Parse a JSON-array column that may arrive as list, str, or None."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(s) for s in raw]
    if isinstance(raw, float) and pd.isna(raw):
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return []
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
    return []


def _build_dim_skill_and_bridge(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    skill_rows: dict[str, dict[str, Any]] = {}
    bridge_rows: set[tuple[str, str]] = set()
    for _, r in df.iterrows():
        job_id = r["job_id"]
        for skill_type, col in (
            (SKILL_TYPE_TECH, "tech_skills"),
            (SKILL_TYPE_SOFT, "soft_skills"),
        ):
            for name in _parse_skills(r[col]):
                n_norm = name.strip().lower()
                if not n_norm:
                    continue
                key = stable_key(n_norm, skill_type)
                if key not in skill_rows:
                    skill_rows[key] = {
                        "skill_key": key,
                        "name": n_norm,
                        "skill_type": skill_type,
                    }
                bridge_rows.add((job_id, key))

    dim_skill = pd.DataFrame(
        list(skill_rows.values()), columns=["skill_key", "name", "skill_type"]
    )
    bridge = pd.DataFrame(
        [{"job_id": j, "skill_key": k} for j, k in bridge_rows],
        columns=["job_id", "skill_key"],
    )
    return dim_skill, bridge


def _build_fact(
    df: pd.DataFrame,
    dim_loc: pd.DataFrame,
    dim_sal: pd.DataFrame,
) -> pd.DataFrame:
    # Map natural key → surrogate for each row.
    rows = []
    for _, r in df.iterrows():
        loc_key = stable_key(_norm(r["city"]), _norm(r["country_code"]))
        smin = r["salary_min"]
        smax = r["salary_max"]
        sal_key = stable_key(
            None if pd.isna(smin) else float(smin),
            None if pd.isna(smax) else float(smax),
            _norm(r["salary_currency"]),
            _norm(r["salary_period"]),
        )
        rows.append(
            {
                "job_id": r["job_id"],
                "run_id": r["run_id"],
                "scraped_at": r["scraped_at"],
                "enriched_at": r["enriched_at"],
                "site": r["site"],
                "search_keyword": r["search_keyword"],
                "company": r["company"] if not pd.isna(r["company"]) else None,
                "title": r["title"] if not pd.isna(r["title"]) else None,
                "job_url": r["job_url"],
                "date_posted": None if pd.isna(r["date_posted"]) else r["date_posted"],
                "work_mode": r["work_mode"] if _norm(r["work_mode"]) else None,
                "location_key": loc_key,
                "salary_key": sal_key,
            }
        )
    # Silence unused warning: dim frames are referenced for clarity/debugging.
    _ = (dim_loc, dim_sal)
    return pd.DataFrame(
        rows,
        columns=[
            "job_id",
            "run_id",
            "scraped_at",
            "enriched_at",
            "site",
            "search_keyword",
            "company",
            "title",
            "job_url",
            "date_posted",
            "work_mode",
            "location_key",
            "salary_key",
        ],
    )


# ---- upsert + marts ------------------------------------------------------- #
def _upsert(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    register_name: str,
    sql_file: str,
) -> int:
    """Register `df`, run the DML file that references it, return row count."""
    if df.empty:
        return 0
    con.register(register_name, df)
    try:
        con.execute(load_sql(sql_file, kind="dml"))
    finally:
        con.unregister(register_name)
    return len(df)


def _refresh_marts(con: duckdb.DuckDBPyConnection) -> int:
    """Refresh all marts atomically so the dashboard never observes an
    empty mart between DELETE and INSERT."""
    count = 0
    con.execute("BEGIN TRANSACTION")
    try:
        for name in _MART_FILES:
            log.info("transform.mart.refresh", file=name)
            con.execute(load_sql(name, kind="dml"))
            count += 1
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return count
