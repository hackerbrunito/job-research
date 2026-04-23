"""Shared helpers for the Streamlit dashboard pages.

Pure-Python helpers (DB queries, profile CRUD) are kept testable without a
running Streamlit session. Streamlit-only wrappers live at the bottom and
are gated by a try/except import.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
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

# Defaults seeded into a first-run profile.
_DEFAULT_KEYWORDS: Final[tuple[str, ...]] = ("Data Engineer",)
_DEFAULT_LOCATIONS: Final[tuple[str, ...]] = ("London, UK",)


# --------------------------------------------------------------------------- #
# Profile model
# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    profile_id: str
    name: str
    description: str | None = None
    keywords: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=lambda: list(C.DEFAULT_SITES))
    created_at: datetime | None = None
    updated_at: datetime | None = None


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Human-readable name → url-safe id. Empty input raises."""
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not s:
        raise ValueError("profile name must contain at least one alphanumeric char")
    return s


def _parse_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    return []


def _row_to_profile(row: tuple) -> Profile:
    return Profile(
        profile_id=row[0],
        name=row[1],
        description=row[2],
        keywords=_parse_list(row[3]),
        locations=_parse_list(row[4]),
        sites=_parse_list(row[5]),
        created_at=row[6],
        updated_at=row[7],
    )


# --------------------------------------------------------------------------- #
# Profile CRUD
# --------------------------------------------------------------------------- #
def list_profiles(con: duckdb.DuckDBPyConnection) -> list[Profile]:
    """Return every saved profile ordered by updated_at desc."""
    rows = con.execute(
        """
        SELECT profile_id, name, description, keywords, locations, sites,
               created_at, updated_at
        FROM user_search_profiles
        ORDER BY updated_at DESC
        """
    ).fetchall()
    return [_row_to_profile(r) for r in rows]


def get_profile(con: duckdb.DuckDBPyConnection, profile_id: str) -> Profile | None:
    row = con.execute(
        """
        SELECT profile_id, name, description, keywords, locations, sites,
               created_at, updated_at
        FROM user_search_profiles
        WHERE profile_id = ?
        """,
        [profile_id],
    ).fetchone()
    return _row_to_profile(row) if row else None


def save_profile(con: duckdb.DuckDBPyConnection, profile: Profile) -> None:
    """Insert or update a profile atomically. Rejects empty name/keywords."""
    name = profile.name.strip()
    if not name:
        raise ValueError("profile name cannot be blank")

    keywords = [k.strip() for k in profile.keywords if k.strip()]
    if not keywords:
        raise ValueError("profile must have at least one keyword")

    locations = [loc.strip() for loc in profile.locations if loc.strip()]
    sites = [s for s in profile.sites if s in C.ALL_SITES]
    if not sites:
        sites = list(C.DEFAULT_SITES)

    pid = profile.profile_id or slugify(name)

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM user_search_profiles WHERE profile_id = ?", [pid])
        con.execute(
            """
            INSERT INTO user_search_profiles (
                profile_id, name, description, keywords, locations, sites,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            [
                pid,
                name,
                (profile.description or "").strip() or None,
                json.dumps(keywords),
                json.dumps(locations),
                json.dumps(sites),
            ],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    log.info(
        "profile.saved",
        profile_id=pid,
        keywords=len(keywords),
        locations=len(locations),
        sites=len(sites),
    )


def delete_profile(con: duckdb.DuckDBPyConnection, profile_id: str) -> None:
    """Delete a profile. Does NOT cascade to fact_job_offers — rows stay
    and can still be inspected by run_id, but lose their profile filter."""
    con.execute("DELETE FROM user_search_profiles WHERE profile_id = ?", [profile_id])
    log.info("profile.deleted", profile_id=profile_id)


def ensure_default_profile(con: duckdb.DuckDBPyConnection) -> Profile:
    """Create a 'default' profile if none exist; seed from legacy
    `user_search_config` if it has data. Idempotent."""
    existing = list_profiles(con)
    if existing:
        return existing[0]

    # Migrate from the legacy single-row table if anything's there.
    legacy: dict[str, list[str]] = {}
    try:
        rows = con.execute("SELECT key, value FROM user_search_config").fetchall()
    except duckdb.Error:
        rows = []
    for key, raw in rows:
        legacy[key] = _parse_list(raw)

    profile = Profile(
        profile_id=C.DEFAULT_PROFILE_ID,
        name=C.DEFAULT_PROFILE_NAME,
        keywords=legacy.get("keywords") or list(_DEFAULT_KEYWORDS),
        locations=legacy.get("locations") or list(_DEFAULT_LOCATIONS),
        sites=legacy.get("sites") or list(C.DEFAULT_SITES),
    )
    save_profile(con, profile)
    return profile


def create_profile_from_name(
    con: duckdb.DuckDBPyConnection,
    name: str,
    *,
    keywords: list[str] | None = None,
    locations: list[str] | None = None,
    sites: list[str] | None = None,
) -> Profile:
    """Shortcut: build a Profile, save it, return it."""
    pid = slugify(name)
    # Ensure uniqueness if a profile with the same id already exists.
    while get_profile(con, pid) is not None:
        pid = f"{slugify(name)}-{uuid.uuid4().hex[:4]}"
    profile = Profile(
        profile_id=pid,
        name=name.strip(),
        keywords=list(keywords or _DEFAULT_KEYWORDS),
        locations=list(locations or _DEFAULT_LOCATIONS),
        sites=list(sites or C.DEFAULT_SITES),
    )
    save_profile(con, profile)
    return profile


# --------------------------------------------------------------------------- #
# DB readers (profile-aware)
# --------------------------------------------------------------------------- #
_MART_TABLES: Final[frozenset[str]] = frozenset(
    {
        "mart_jobs_by_country",
        "mart_skills_by_keyword",
        "mart_salary_by_keyword",
        "mart_work_mode_distribution",
    }
)


def load_mart(
    con: duckdb.DuckDBPyConnection,
    name: str,
    *,
    profile_id: str | None = None,
) -> pd.DataFrame:
    """Load a mart table as a DataFrame. Optional profile filter."""
    if name not in _MART_TABLES:
        raise ValueError(f"Unknown mart {name!r}. Allowed: {sorted(_MART_TABLES)}")
    if profile_id is None:
        return con.execute(f"SELECT * FROM {name}").df()
    return con.execute(
        f"SELECT * FROM {name} WHERE profile_id = ?",
        [profile_id],
    ).df()


def load_fact_with_dims(
    con: duckdb.DuckDBPyConnection,
    *,
    keyword: str | None = None,
    profile_id: str | None = None,
    limit: int = DEFAULT_FACT_LIMIT,
) -> pd.DataFrame:
    """Join fact_job_offers with dim_location, dim_salary, int_enriched_job_info."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    limit = min(limit, MAX_FACT_LIMIT)

    sql = """
        SELECT
            f.job_id,
            f.run_id,
            f.profile_id,
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
            e.domain_skills,
            f.scraped_at,
            f.enriched_at
        FROM fact_job_offers f
        LEFT JOIN dim_location          l ON l.location_key = f.location_key
        LEFT JOIN dim_salary            s ON s.salary_key   = f.salary_key
        LEFT JOIN int_enriched_job_info e ON e.job_id       = f.job_id
    """
    params: list[Any] = []
    clauses: list[str] = []
    if keyword:
        clauses.append("f.search_keyword = ?")
        params.append(keyword)
    if profile_id:
        clauses.append("f.profile_id = ?")
        params.append(profile_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY f.date_posted DESC NULLS LAST, f.scraped_at DESC LIMIT ?"
    params.append(limit)

    return con.execute(sql, params).df()


def count_jobs(con: duckdb.DuckDBPyConnection, *, profile_id: str | None = None) -> int:
    """Rows in fact_job_offers (optionally filtered by profile)."""
    try:
        if profile_id is None:
            result = con.execute("SELECT COUNT(*) FROM fact_job_offers").fetchone()
        else:
            result = con.execute(
                "SELECT COUNT(*) FROM fact_job_offers WHERE profile_id = ?",
                [profile_id],
            ).fetchone()
    except duckdb.Error:
        return 0
    return int(result[0]) if result else 0


def latest_run_status(
    con: duckdb.DuckDBPyConnection, *, profile_id: str | None = None
) -> dict[str, Any] | None:
    """Return {run_id, started_at, finished_at, status, profile_id} for the
    most recent run (optionally filtered)."""
    try:
        if profile_id is None:
            row = con.execute(
                """
                SELECT run_id, started_at, finished_at, status, profile_id
                FROM pipeline_runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()
        else:
            row = con.execute(
                """
                SELECT run_id, started_at, finished_at, status, profile_id
                FROM pipeline_runs
                WHERE profile_id = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                [profile_id],
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
        "profile_id": row[4],
    }


# ---- triage label CRUD --------------------------------------------------- #


def _norm_title(title: str | None) -> str:
    """Lowercase + strip for label lookup key."""
    return (title or "").strip().lower()


def list_title_labels(
    con: duckdb.DuckDBPyConnection, profile_id: str
) -> list[dict[str, Any]]:
    """Return all labels for a profile, ordered by count_seen desc."""
    rows = con.execute(
        """
        SELECT title_norm, label, note, count_seen, updated_at
        FROM profile_title_labels
        WHERE profile_id = ?
        ORDER BY count_seen DESC, title_norm
        """,
        [profile_id],
    ).fetchall()
    return [
        {
            "title_norm": r[0],
            "label": r[1],
            "note": r[2],
            "count_seen": r[3],
            "updated_at": r[4],
        }
        for r in rows
    ]


def save_title_label(
    con: duckdb.DuckDBPyConnection,
    *,
    profile_id: str,
    title_norm: str,
    label: str,  # 'accept' | 'reject' | 'unsure'
    note: str | None = None,
    count_seen: int = 1,
) -> None:
    """Upsert a single label. Increments count_seen on conflict."""
    if label not in {"accept", "reject", "unsure"}:
        raise ValueError(f"label must be accept/reject/unsure, got {label!r}")
    con.execute(
        """
        INSERT INTO profile_title_labels
            (profile_id, title_norm, label, note, count_seen, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, now(), now())
        ON CONFLICT (profile_id, title_norm) DO UPDATE SET
            label      = excluded.label,
            note       = excluded.note,
            count_seen = profile_title_labels.count_seen + 1,
            updated_at = now()
        """,
        [profile_id, title_norm, label, note, count_seen],
    )


def delete_title_label(
    con: duckdb.DuckDBPyConnection, *, profile_id: str, title_norm: str
) -> None:
    con.execute(
        "DELETE FROM profile_title_labels WHERE profile_id = ? AND title_norm = ?",
        [profile_id, title_norm],
    )


def apply_title_labels_to_judged(
    con: duckdb.DuckDBPyConnection, profile_id: str
) -> int:
    """Update judged_job_offers.ensemble_verdict for rows whose title matches
    a user-confirmed label for this profile.

    Returns the number of rows updated.
    """
    # Accept labels → override to 'accept' (even if rule said review)
    con.execute(
        """
        UPDATE judged_job_offers
        SET ensemble_verdict = 'accept',
            judged_at        = CURRENT_TIMESTAMP
        WHERE profile_id = ?
          AND lower(trim(job_title)) IN (
              SELECT title_norm FROM profile_title_labels
              WHERE profile_id = ? AND label = 'accept'
          )
          AND ensemble_verdict != 'accept'
        """,
        [profile_id, profile_id],
    )
    # Reject labels → override to 'reject'
    con.execute(
        """
        UPDATE judged_job_offers
        SET ensemble_verdict = 'reject',
            judged_at        = CURRENT_TIMESTAMP
        WHERE profile_id = ?
          AND lower(trim(job_title)) IN (
              SELECT title_norm FROM profile_title_labels
              WHERE profile_id = ? AND label = 'reject'
          )
          AND ensemble_verdict != 'reject'
        """,
        [profile_id, profile_id],
    )
    # Return total rows now with a user-label-driven verdict.
    result = con.execute(
        """
        SELECT COUNT(*) FROM judged_job_offers j
        JOIN profile_title_labels l
          ON j.profile_id = l.profile_id
         AND lower(trim(j.job_title)) = l.title_norm
        WHERE j.profile_id = ?
        """,
        [profile_id],
    ).fetchone()
    return int(result[0]) if result else 0


def get_triage_candidates(
    con: duckdb.DuckDBPyConnection,
    profile_id: str,
    *,
    include_decided: bool = False,
) -> pd.DataFrame:
    """Return unique job titles from staging for this profile, with verdict info.

    Columns: title_norm, display_title, company_sample, count,
             rule_verdict, ensemble_verdict, user_label.

    include_decided=False (default) shows only rows where user hasn't
    labelled yet (unlabelled or unsure). True shows all.
    """
    label_filter = (
        "" if include_decided else "AND (l.label IS NULL OR l.label = 'unsure')"
    )
    sql = f"""
        SELECT
            lower(trim(j.job_title))                             AS title_norm,
            j.job_title                                          AS display_title,
            any_value(s.company)                                 AS company_sample,
            COUNT(*)                                             AS count,
            any_value(j.rule_verdict)                            AS rule_verdict,
            any_value(j.ensemble_verdict)                        AS ensemble_verdict,
            any_value(l.label)                                   AS user_label
        FROM judged_job_offers j
        JOIN staging_job_offers s ON s.id = j.job_id
        LEFT JOIN profile_title_labels l
               ON l.profile_id = j.profile_id
              AND l.title_norm  = lower(trim(j.job_title))
        WHERE j.profile_id = ?
        {label_filter}
        GROUP BY lower(trim(j.job_title)), j.job_title
        ORDER BY count DESC, title_norm
        LIMIT 200
    """
    return con.execute(sql, [profile_id]).df()


# --------------------------------------------------------------------------- #
# Connection helper
# --------------------------------------------------------------------------- #
@contextmanager
def read_only_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the shared DuckDB read-only. Initializes schema on first open."""
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
    def cached_mart(name: str, profile_id: str | None = None) -> pd.DataFrame:
        with read_only_connection() as con:
            return load_mart(con, name, profile_id=profile_id)

    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def cached_fact_with_dims(
        keyword: str | None = None,
        profile_id: str | None = None,
        limit: int = DEFAULT_FACT_LIMIT,
    ) -> pd.DataFrame:
        with read_only_connection() as con:
            return load_fact_with_dims(
                con, keyword=keyword, profile_id=profile_id, limit=limit
            )

    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def cached_count_jobs(profile_id: str | None = None) -> int:
        with read_only_connection() as con:
            return count_jobs(con, profile_id=profile_id)

    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def cached_latest_run_status(
        profile_id: str | None = None,
    ) -> dict[str, Any] | None:
        with read_only_connection() as con:
            return latest_run_status(con, profile_id=profile_id)

except ImportError:  # pragma: no cover — tests don't need this branch
    pass
