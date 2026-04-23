"""Smoke tests for database.py: schema applies cleanly, helpers work."""

from __future__ import annotations

import duckdb

from job_research.database import job_id, stable_key

EXPECTED_TABLES = {
    "staging_job_offers",
    "int_enriched_job_info",
    "dim_location",
    "dim_salary",
    "dim_skill",
    "fact_job_offers",
    "job_skill_bridge",
    "mart_jobs_by_country",
    "mart_skills_by_keyword",
    "mart_salary_by_keyword",
    "mart_work_mode_distribution",
    "pipeline_runs",
    "user_search_config",
    "user_search_profiles",
}


def test_schema_creates_all_tables(tmp_duckdb: duckdb.DuckDBPyConnection) -> None:
    rows = tmp_duckdb.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    tables = {r[0] for r in rows}
    missing = EXPECTED_TABLES - tables
    assert not missing, f"missing tables: {sorted(missing)}"


def test_stable_key_is_deterministic() -> None:
    a = stable_key("London", "GB")
    b = stable_key("London", "GB")
    c = stable_key("London", "US")
    assert a == b
    assert a != c
    assert len(a) == 24  # blake2b digest_size=12 hex


def test_job_id_survives_resample() -> None:
    assert job_id("linkedin", "https://x/1") == job_id("linkedin", "https://x/1")
    assert job_id("linkedin", "https://x/1") != job_id("indeed", "https://x/1")


def test_pipeline_runs_insert(tmp_duckdb: duckdb.DuckDBPyConnection) -> None:
    tmp_duckdb.execute(
        "INSERT INTO pipeline_runs (run_id, started_at, status) "
        "VALUES (?, CURRENT_TIMESTAMP, 'running')",
        ["r1"],
    )
    status = tmp_duckdb.execute(
        "SELECT status FROM pipeline_runs WHERE run_id = ?", ["r1"]
    ).fetchone()
    assert status is not None and status[0] == "running"
