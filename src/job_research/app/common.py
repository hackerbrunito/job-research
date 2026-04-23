"""Shared helpers for the Streamlit dashboard pages.

The module is deliberately split so that pure-Python helpers (DB queries,
config round-tripping) can be unit-tested without a running Streamlit
session. The Streamlit-only wrappers are defined at the bottom and gated
by a try/except import.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

import duckdb
import pandas as pd

from job_research import constants as C
from job_research.database import connect, init_schema
from job_research.logging_setup import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Named constants (no magic numbers in page logic)
# --------------------------------------------------------------------------- #
DEFAULT_FACT_LIMIT: Final[int] = 500
MAX_FACT_LIMIT: Final[int] = 5000
CACHE_TTL_SECONDS: Final[int] = 60

CONFIG_KEY_KEYWORDS: Final[str] = "keywords"
CONFIG_KEY_LOCATIONS: Final[str] = "locations"
CONFIG_KEY_SITES: Final[str] = "sites"

_CONFIG_KEYS: Final[tuple[str, ...]] = (
    CONFIG_KEY_KEYWORDS,
    CONFIG_KEY_LOCATIONS,
    CONFIG_KEY_SITES,
)

# Sensible defaults when the table is empty.
_DEFAULT_KEYWORDS: Final[tuple[str, ...]] = ("AI Cybersecurity", "Data Engineer")
_DEFAULT_LOCATIONS: Final[tuple[str, ...]] = ("London, UK", "Berlin, Germany")


# --------------------------------------------------------------------------- #
# Config round-tripping
# --------------------------------------------------------------------------- #
def get_search_config(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """Read user_search_config into a {keywords, locations, sites} dict.

    Returns sensible defaults if the table is empty or any key is missing.
    """
    rows = con.execute(
        "SELECT key, value FROM user_search_config WHERE key IN (?, ?, ?)",
        list(_CONFIG_KEYS),
    ).fetchall()

    stored: dict[str, list[str]] = {}
    for key, raw in rows:
        parsed = raw if isinstance(raw, list) else json.loads(raw)
        stored[key] = [str(x) for x in parsed]

    return {
        CONFIG_KEY_KEYWORDS: stored.get(CONFIG_KEY_KEYWORDS, list(_DEFAULT_KEYWORDS)),
        CONFIG_KEY_LOCATIONS: stored.get(
            CONFIG_KEY_LOCATIONS, list(_DEFAULT_LOCATIONS)
        ),
        CONFIG_KEY_SITES: stored.get(CONFIG_KEY_SITES, list(C.DEFAULT_SITES)),
    }


def save_search_config(
    con: duckdb.DuckDBPyConnection,
    *,
    keywords: list[str],
    locations: list[str],
    sites: list[str],
) -> None:
    """Upsert all three keys atomically."""
    payload: dict[str, list[str]] = {
        CONFIG_KEY_KEYWORDS: [k.strip() for k in keywords if k.strip()],
        CONFIG_KEY_LOCATIONS: [loc.strip() for loc in locations if loc.strip()],
        CONFIG_KEY_SITES: [s for s in sites if s in C.ALL_SITES],
    }

    con.execute("BEGIN TRANSACTION")
    try:
        for key, value in payload.items():
            con.execute(
                "DELETE FROM user_search_config WHERE key = ?",
                [key],
            )
            con.execute(
                """
                INSERT INTO user_search_config (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                [key, json.dumps(value)],
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    log.info(
        "user_config.saved",
        keywords=len(payload[CONFIG_KEY_KEYWORDS]),
        locations=len(payload[CONFIG_KEY_LOCATIONS]),
        sites=len(payload[CONFIG_KEY_SITES]),
    )


# --------------------------------------------------------------------------- #
# DB readers
# --------------------------------------------------------------------------- #
_MART_TABLES: Final[frozenset[str]] = frozenset(
    {
        "mart_jobs_by_country",
        "mart_skills_by_keyword",
        "mart_salary_by_keyword",
        "mart_work_mode_distribution",
    }
)


def load_mart(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Load a mart table as a DataFrame. Validates the name against an allow-list."""
    if name not in _MART_TABLES:
        raise ValueError(f"Unknown mart {name!r}. Allowed: {sorted(_MART_TABLES)}")
    return con.execute(f"SELECT * FROM {name}").df()


def load_fact_with_dims(
    con: duckdb.DuckDBPyConnection,
    *,
    keyword: str | None = None,
    limit: int = DEFAULT_FACT_LIMIT,
) -> pd.DataFrame:
    """Join fact_job_offers with dim_location, dim_salary, int_enriched_job_info.

    Returns a DataFrame suitable for direct rendering in st.dataframe.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    limit = min(limit, MAX_FACT_LIMIT)

    sql = """
        SELECT
            f.job_id,
            f.run_id,
            f.site,
            f.search_keyword,
            f.company,
            f.title,
            f.job_url,
            f.date_posted,
            f.work_mode,
            l.city,
            l.country,
            l.country_code,
            s.min_amount AS salary_min,
            s.max_amount AS salary_max,
            s.currency   AS salary_currency,
            s.period     AS salary_period,
            e.tech_skills,
            e.soft_skills,
            f.scraped_at,
            f.enriched_at
        FROM fact_job_offers f
        LEFT JOIN dim_location          l ON l.location_key = f.location_key
        LEFT JOIN dim_salary            s ON s.salary_key   = f.salary_key
        LEFT JOIN int_enriched_job_info e ON e.job_id       = f.job_id
    """
    params: list[Any] = []
    if keyword:
        sql += " WHERE f.search_keyword = ?"
        params.append(keyword)
    sql += " ORDER BY f.date_posted DESC NULLS LAST, f.scraped_at DESC LIMIT ?"
    params.append(limit)

    return con.execute(sql, params).df()


def count_jobs(con: duckdb.DuckDBPyConnection) -> int:
    """Quick count of rows in fact_job_offers (0 if table missing/empty)."""
    try:
        result = con.execute("SELECT COUNT(*) FROM fact_job_offers").fetchone()
    except duckdb.Error:
        return 0
    return int(result[0]) if result else 0


def latest_run_status(con: duckdb.DuckDBPyConnection) -> dict[str, Any] | None:
    """Return {run_id, started_at, finished_at, status} for the most recent run."""
    try:
        row = con.execute(
            """
            SELECT run_id, started_at, finished_at, status
            FROM pipeline_runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()
    except duckdb.Error:
        return None
    if row is None:
        return None
    return {
        "run_id": row[0],
        "started_at": row[1],
        "finished_at": row[2],
        "status": row[3],
    }


# --------------------------------------------------------------------------- #
# Connection helper
# --------------------------------------------------------------------------- #
@contextmanager
def read_only_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the shared DuckDB read-only.

    If the DB file doesn't exist yet, init_schema() first (creates it
    read-write) and then reopen read-only.
    """
    from job_research.config import get_settings

    settings = get_settings()
    if not settings.database.path.is_file():
        log.info("database.init_on_first_open", path=str(settings.database.path))
        init_schema()

    with connect(read_only=True) as con:
        yield con


# --------------------------------------------------------------------------- #
# Streamlit-only cached wrappers
# --------------------------------------------------------------------------- #
try:
    import streamlit as st

    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def cached_mart(name: str) -> pd.DataFrame:
        """Cached version of load_mart for use inside Streamlit pages."""
        with read_only_connection() as con:
            return load_mart(con, name)

    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def cached_fact_with_dims(
        keyword: str | None = None, limit: int = DEFAULT_FACT_LIMIT
    ) -> pd.DataFrame:
        """Cached version of load_fact_with_dims."""
        with read_only_connection() as con:
            return load_fact_with_dims(con, keyword=keyword, limit=limit)

    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def cached_count_jobs() -> int:
        with read_only_connection() as con:
            return count_jobs(con)

    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def cached_latest_run_status() -> dict[str, Any] | None:
        with read_only_connection() as con:
            return latest_run_status(con)

except ImportError:  # pragma: no cover — tests don't need this branch
    pass
