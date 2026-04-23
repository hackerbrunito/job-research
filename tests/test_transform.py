"""Tests for the transform layer: dims + facts + bridge + marts."""

from __future__ import annotations

import json
from datetime import date, datetime

import duckdb
import pytest

from job_research.database import job_id
from job_research.transform import run_transform


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def seeded_duckdb(tmp_duckdb: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Seed 4 overlapping staging + enriched rows for transform tests."""
    run_id = "r-test"
    rows = [
        {
            "id": job_id("linkedin", "https://example.com/1"),
            "site": "linkedin",
            "search_keyword": "data engineer",
            "search_location": "London",
            "job_url": "https://example.com/1",
            "title": "Senior Data Engineer",
            "company": "Acme",
            "date_posted": date(2026, 4, 1),
            "city": "London",
            "country": "United Kingdom",
            "country_code": "GB",
            "work_mode": "remote",
            "tech": ["python", "sql"],
            "soft": ["communicative"],
            "salary_min": 80000.0,
            "salary_max": 100000.0,
            "currency": "GBP",
            "period": "yearly",
        },
        {
            "id": job_id("linkedin", "https://example.com/2"),
            "site": "linkedin",
            "search_keyword": "data engineer",
            "search_location": "London",
            "job_url": "https://example.com/2",
            "title": "Data Engineer",
            "company": "Beta",
            "date_posted": date(2026, 4, 2),
            "city": "London",  # shared location
            "country": "United Kingdom",
            "country_code": "GB",
            "work_mode": "hybrid",
            "tech": ["python", "airflow"],  # python shared with row 1
            "soft": ["collaborative"],
            "salary_min": 80000.0,  # shared salary bucket with row 1
            "salary_max": 100000.0,
            "currency": "GBP",
            "period": "yearly",
        },
        {
            "id": job_id("indeed", "https://example.com/3"),
            "site": "indeed",
            "search_keyword": "ml engineer",
            "search_location": "Berlin",
            "job_url": "https://example.com/3",
            "title": "ML Engineer",
            "company": "Gamma",
            "date_posted": date(2026, 4, 3),
            "city": "Berlin",  # different location
            "country": "Germany",
            "country_code": "DE",
            "work_mode": "on-site",
            "tech": ["python", "pytorch"],  # python shared across keywords
            "soft": ["analytical"],
            "salary_min": 70000.0,
            "salary_max": 90000.0,
            "currency": "EUR",
            "period": "yearly",
        },
        {
            "id": job_id("indeed", "https://example.com/4"),
            "site": "indeed",
            "search_keyword": "ml engineer",
            "search_location": "Berlin",
            "job_url": "https://example.com/4",
            "title": "Senior ML Engineer",
            "company": "Delta",
            "date_posted": date(2026, 4, 4),
            "city": "Berlin",
            "country": "Germany",
            "country_code": "DE",
            "work_mode": "remote",
            "tech": ["pytorch", "kubernetes"],
            "soft": ["analytical"],
            "salary_min": 90000.0,
            "salary_max": 110000.0,
            "currency": "EUR",
            "period": "yearly",
        },
    ]

    now = datetime(2026, 4, 23, 12, 0, 0)
    for r in rows:
        tmp_duckdb.execute(
            """
            INSERT INTO staging_job_offers (
                id, scraped_at, run_id, site, search_keyword, search_location,
                job_url, title, company, date_posted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                r["id"],
                now,
                run_id,
                r["site"],
                r["search_keyword"],
                r["search_location"],
                r["job_url"],
                r["title"],
                r["company"],
                r["date_posted"],
            ],
        )
        tmp_duckdb.execute(
            """
            INSERT INTO int_enriched_job_info (
                job_id, enriched_at, llm_provider, llm_model,
                tech_skills, soft_skills, city, country, country_code,
                work_mode, salary_min, salary_max, salary_currency, salary_period
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                r["id"],
                now,
                "anthropic",
                "claude-haiku-4-5-20251001",
                json.dumps(r["tech"]),
                json.dumps(r["soft"]),
                r["city"],
                r["country"],
                r["country_code"],
                r["work_mode"],
                r["salary_min"],
                r["salary_max"],
                r["currency"],
                r["period"],
            ],
        )
    return tmp_duckdb


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_transform_populates_dims_fact_bridge(
    seeded_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    summary = run_transform(con=seeded_duckdb)

    # dim_location — two distinct (city, country_code): London/GB, Berlin/DE.
    assert summary.dim_location_rows == 2
    loc_rows = seeded_duckdb.execute(
        "SELECT city, country_code FROM dim_location ORDER BY city"
    ).fetchall()
    assert set(loc_rows) == {("Berlin", "DE"), ("London", "GB")}

    # dim_salary — three distinct buckets: (80k,100k,GBP), (70k,90k,EUR), (90k,110k,EUR).
    assert summary.dim_salary_rows == 3

    # dim_skill — deduplicated across jobs:
    # tech: python, sql, airflow, pytorch, kubernetes = 5
    # soft: communicative, collaborative, analytical = 3
    assert summary.dim_skill_rows == 8

    # fact — one per job.
    assert summary.fact_rows == 4
    fact_count = seeded_duckdb.execute(
        "SELECT COUNT(*) FROM fact_job_offers"
    ).fetchone()
    assert fact_count is not None and fact_count[0] == 4

    # bridge — sum of skills per job (2+1)+(2+1)+(2+1)+(2+1) = 12.
    bridge_count = seeded_duckdb.execute(
        "SELECT COUNT(*) FROM job_skill_bridge"
    ).fetchone()
    assert bridge_count is not None and bridge_count[0] == 12

    # Python skill is shared across 3 jobs.
    python_jobs = seeded_duckdb.execute(
        """
        SELECT COUNT(*) FROM job_skill_bridge b
        JOIN dim_skill s ON b.skill_key = s.skill_key
        WHERE s.name = 'python' AND s.skill_type = 'tech'
        """
    ).fetchone()
    assert python_jobs is not None and python_jobs[0] == 3


def test_transform_populates_marts(
    seeded_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    summary = run_transform(con=seeded_duckdb)
    assert summary.marts_refreshed == 4

    # mart_jobs_by_country: 2 (GB, data engineer), 2 (DE, ml engineer).
    country_rows = seeded_duckdb.execute(
        "SELECT country_code, search_keyword, job_count FROM mart_jobs_by_country "
        "ORDER BY country_code"
    ).fetchall()
    assert country_rows == [
        ("DE", "ml engineer", 2),
        ("GB", "data engineer", 2),
    ]

    # mart_skills_by_keyword lists seeded skills.
    skills = seeded_duckdb.execute(
        "SELECT skill_name FROM mart_skills_by_keyword"
    ).fetchall()
    names = {r[0] for r in skills}
    assert {
        "python",
        "sql",
        "airflow",
        "pytorch",
        "kubernetes",
        "communicative",
        "collaborative",
        "analytical",
    } <= names

    # mart_salary_by_keyword — percentile columns filled, not null.
    sal = seeded_duckdb.execute(
        """
        SELECT search_keyword, currency, p25_min, p50_min, p75_min,
               p25_max, p50_max, p75_max, observation_count
        FROM mart_salary_by_keyword
        ORDER BY search_keyword, currency
        """
    ).fetchall()
    assert len(sal) == 2  # (data engineer, GBP), (ml engineer, EUR)
    for row in sal:
        # Percentile cols at indices 2..7 must be non-null.
        for v in row[2:8]:
            assert v is not None

    # data engineer / GBP: both rows are 80k-100k so all percentiles == 80000 / 100000.
    de = seeded_duckdb.execute(
        "SELECT p25_min, p50_min, p75_min, p25_max, p50_max, p75_max "
        "FROM mart_salary_by_keyword WHERE search_keyword = 'data engineer'"
    ).fetchone()
    assert de == (80000.0, 80000.0, 80000.0, 100000.0, 100000.0, 100000.0)

    # mart_work_mode_distribution — counts sum to total fact rows.
    total_mode = seeded_duckdb.execute(
        "SELECT SUM(job_count) FROM mart_work_mode_distribution"
    ).fetchone()
    assert total_mode is not None and total_mode[0] == 4


def test_transform_is_idempotent(
    seeded_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    run_transform(con=seeded_duckdb)
    first = {
        "dim_location": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM dim_location"
        ).fetchone()[0],
        "dim_salary": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM dim_salary"
        ).fetchone()[0],
        "dim_skill": seeded_duckdb.execute("SELECT COUNT(*) FROM dim_skill").fetchone()[
            0
        ],
        "fact": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM fact_job_offers"
        ).fetchone()[0],
        "bridge": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM job_skill_bridge"
        ).fetchone()[0],
        "mart_country": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM mart_jobs_by_country"
        ).fetchone()[0],
        "mart_skills": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM mart_skills_by_keyword"
        ).fetchone()[0],
        "mart_salary": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM mart_salary_by_keyword"
        ).fetchone()[0],
        "mart_mode": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM mart_work_mode_distribution"
        ).fetchone()[0],
    }

    # Re-run — counts must not double.
    run_transform(con=seeded_duckdb)
    second = {
        "dim_location": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM dim_location"
        ).fetchone()[0],
        "dim_salary": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM dim_salary"
        ).fetchone()[0],
        "dim_skill": seeded_duckdb.execute("SELECT COUNT(*) FROM dim_skill").fetchone()[
            0
        ],
        "fact": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM fact_job_offers"
        ).fetchone()[0],
        "bridge": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM job_skill_bridge"
        ).fetchone()[0],
        "mart_country": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM mart_jobs_by_country"
        ).fetchone()[0],
        "mart_skills": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM mart_skills_by_keyword"
        ).fetchone()[0],
        "mart_salary": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM mart_salary_by_keyword"
        ).fetchone()[0],
        "mart_mode": seeded_duckdb.execute(
            "SELECT COUNT(*) FROM mart_work_mode_distribution"
        ).fetchone()[0],
    }
    assert first == second


def test_transform_empty_intermediate_is_safe(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    summary = run_transform(con=tmp_duckdb)
    assert summary.dim_location_rows == 0
    assert summary.fact_rows == 0
    assert summary.marts_refreshed == 4


# --------------------------------------------------------------------------- #
# Verdict-filter tests (Wave 7A)
# --------------------------------------------------------------------------- #
def _seed_one_row(
    con: duckdb.DuckDBPyConnection,
    *,
    jid: str,
    ensemble_verdict: str | None,
) -> None:
    """Insert a minimal staging + enriched row, with optional judged entry."""
    now = datetime(2026, 4, 23, 12, 0, 0)
    con.execute(
        """
        INSERT INTO staging_job_offers (
            id, scraped_at, run_id, site, search_keyword, job_url, title
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            jid,
            now,
            "run-v7",
            "linkedin",
            "python",
            f"https://example.test/{jid}",
            "Dev",
        ],
    )
    con.execute(
        """
        INSERT INTO int_enriched_job_info (
            job_id, enriched_at, llm_provider, llm_model,
            tech_skills, soft_skills
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [jid, now, "fake", "fake-1", json.dumps(["python"]), json.dumps([])],
    )
    if ensemble_verdict is not None:
        con.execute(
            """
            INSERT INTO judged_job_offers
                (job_id, rule_verdict, ensemble_verdict, judged_at)
            VALUES (?, 'accept', ?, CURRENT_TIMESTAMP)
            """,
            [jid, ensemble_verdict],
        )


def test_transform_excludes_rejected_rows(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """Rows with ensemble_verdict='reject' must not appear in fact_job_offers."""
    _seed_one_row(tmp_duckdb, jid="j-reject", ensemble_verdict="reject")
    run_transform(con=tmp_duckdb)
    count = tmp_duckdb.execute("SELECT COUNT(*) FROM fact_job_offers").fetchone()[0]
    assert count == 0


def test_transform_includes_accepted_rows(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """Rows with ensemble_verdict='accept' must appear in fact_job_offers."""
    _seed_one_row(tmp_duckdb, jid="j-accept", ensemble_verdict="accept")
    run_transform(con=tmp_duckdb)
    count = tmp_duckdb.execute("SELECT COUNT(*) FROM fact_job_offers").fetchone()[0]
    assert count == 1


def test_transform_includes_legacy_rows_without_judged_entry(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """Pre-Wave-7A rows (no judged entry) must still flow through to fact."""
    _seed_one_row(tmp_duckdb, jid="j-legacy", ensemble_verdict=None)
    run_transform(con=tmp_duckdb)
    count = tmp_duckdb.execute("SELECT COUNT(*) FROM fact_job_offers").fetchone()[0]
    assert count == 1
