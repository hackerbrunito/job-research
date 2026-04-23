"""Unit tests for pure-Python helpers in job_research.app.common.

These tests never import Streamlit — only the DB helpers are exercised.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pytest

from job_research import constants as C
from job_research.app.common import (
    CONFIG_KEY_KEYWORDS,
    CONFIG_KEY_LOCATIONS,
    CONFIG_KEY_SITES,
    get_search_config,
    load_fact_with_dims,
    load_mart,
    save_search_config,
)
from job_research.database import stable_key


# --------------------------------------------------------------------------- #
# get_search_config / save_search_config
# --------------------------------------------------------------------------- #
def test_get_search_config_defaults_when_empty(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    cfg = get_search_config(tmp_duckdb)

    assert set(cfg.keys()) == {
        CONFIG_KEY_KEYWORDS,
        CONFIG_KEY_LOCATIONS,
        CONFIG_KEY_SITES,
    }
    # All defaults are non-empty lists of strings.
    for key, value in cfg.items():
        assert isinstance(value, list), key
        assert all(isinstance(v, str) for v in value), key
        assert len(value) > 0, key
    # Sites default must be a subset of ALL_SITES.
    assert set(cfg[CONFIG_KEY_SITES]).issubset(set(C.ALL_SITES))


def test_save_then_get_roundtrip(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    save_search_config(
        tmp_duckdb,
        keywords=["Data Engineer", "AI Cybersecurity"],
        locations=["London, UK", "Berlin, Germany"],
        sites=[C.SITE_LINKEDIN, C.SITE_INDEED],
    )

    cfg = get_search_config(tmp_duckdb)
    assert cfg[CONFIG_KEY_KEYWORDS] == ["Data Engineer", "AI Cybersecurity"]
    assert cfg[CONFIG_KEY_LOCATIONS] == ["London, UK", "Berlin, Germany"]
    assert cfg[CONFIG_KEY_SITES] == [C.SITE_LINKEDIN, C.SITE_INDEED]


def test_save_strips_whitespace_and_filters_invalid_sites(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    save_search_config(
        tmp_duckdb,
        keywords=["  Data Engineer  ", "", "  "],
        locations=["London "],
        sites=[C.SITE_LINKEDIN, "not_a_site"],
    )
    cfg = get_search_config(tmp_duckdb)
    assert cfg[CONFIG_KEY_KEYWORDS] == ["Data Engineer"]
    assert cfg[CONFIG_KEY_LOCATIONS] == ["London"]
    assert cfg[CONFIG_KEY_SITES] == [C.SITE_LINKEDIN]


def test_save_upsert_overwrites(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    save_search_config(
        tmp_duckdb,
        keywords=["old"],
        locations=["oldloc"],
        sites=[C.SITE_LINKEDIN],
    )
    save_search_config(
        tmp_duckdb,
        keywords=["new"],
        locations=["newloc"],
        sites=[C.SITE_INDEED],
    )
    cfg = get_search_config(tmp_duckdb)
    assert cfg[CONFIG_KEY_KEYWORDS] == ["new"]
    assert cfg[CONFIG_KEY_LOCATIONS] == ["newloc"]
    assert cfg[CONFIG_KEY_SITES] == [C.SITE_INDEED]


# --------------------------------------------------------------------------- #
# load_mart
# --------------------------------------------------------------------------- #
def test_load_mart_empty_returns_empty_df(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    df = load_mart(tmp_duckdb, "mart_jobs_by_country")
    assert df.empty
    # Columns still present (from the DDL).
    assert "country_code" in df.columns
    assert "job_count" in df.columns


def test_load_mart_with_data(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    tmp_duckdb.execute(
        """
        INSERT INTO mart_jobs_by_country
            (country_code, country, search_keyword, job_count)
        VALUES
            ('GB', 'United Kingdom', 'Data Engineer', 12),
            ('DE', 'Germany',        'Data Engineer',  7)
        """
    )
    df = load_mart(tmp_duckdb, "mart_jobs_by_country")
    assert len(df) == 2
    assert set(df["country_code"]) == {"GB", "DE"}
    assert df["job_count"].sum() == 19


def test_load_mart_rejects_unknown_name(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(ValueError):
        load_mart(tmp_duckdb, "mart_drop_tables")


# --------------------------------------------------------------------------- #
# load_fact_with_dims
# --------------------------------------------------------------------------- #
def _seed_fact_row(
    con: duckdb.DuckDBPyConnection,
    *,
    job_url: str,
    keyword: str,
    country: str,
    country_code: str,
    city: str = "London",
) -> None:
    """Insert a full fact row with its dim + staging + intermediate rows."""
    job_id = stable_key("linkedin", job_url)
    loc_key = stable_key(city.lower(), country_code)
    sal_key = stable_key(100000.0, 150000.0, "USD", "yearly")
    now = datetime.now(UTC).replace(tzinfo=None)

    con.execute(
        """
        INSERT INTO staging_job_offers
            (id, scraped_at, run_id, site, search_keyword, job_url)
        VALUES (?, ?, 'r1', 'linkedin', ?, ?)
        """,
        [job_id, now, keyword, job_url],
    )
    con.execute(
        "INSERT OR IGNORE INTO dim_location VALUES (?, ?, ?, ?)",
        [loc_key, city, country, country_code],
    )
    con.execute(
        "INSERT OR IGNORE INTO dim_salary VALUES (?, ?, ?, ?, ?)",
        [sal_key, 100000.0, 150000.0, "USD", "yearly"],
    )
    con.execute(
        """
        INSERT INTO int_enriched_job_info (
            job_id, enriched_at, llm_provider, llm_model,
            tech_skills, soft_skills,
            city, country, country_code, work_mode,
            salary_min, salary_max, salary_currency, salary_period
        )
        VALUES (?, ?, 'anthropic', 'claude-haiku-4-5',
                ?, ?, ?, ?, ?, 'remote',
                100000.0, 150000.0, 'USD', 'yearly')
        """,
        [
            job_id,
            now,
            '["python","sql"]',
            '["collaborative"]',
            city,
            country,
            country_code,
        ],
    )
    con.execute(
        """
        INSERT INTO fact_job_offers (
            job_id, run_id, scraped_at, enriched_at, site,
            search_keyword, company, title, job_url, date_posted,
            work_mode, location_key, salary_key
        )
        VALUES (?, 'r1', ?, ?, 'linkedin',
                ?, 'Acme', 'Data Engineer', ?, DATE '2026-04-20',
                'remote', ?, ?)
        """,
        [job_id, now, now, keyword, job_url, loc_key, sal_key],
    )


def test_load_fact_with_dims_empty(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    df = load_fact_with_dims(tmp_duckdb)
    assert df.empty
    expected_cols = {
        "job_id",
        "site",
        "search_keyword",
        "city",
        "country",
        "salary_min",
        "salary_currency",
        "tech_skills",
    }
    assert expected_cols.issubset(set(df.columns))


def test_load_fact_with_dims_joins_and_respects_keyword_filter(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    _seed_fact_row(
        tmp_duckdb,
        job_url="https://example.com/j/1",
        keyword="Data Engineer",
        country="United Kingdom",
        country_code="GB",
    )
    _seed_fact_row(
        tmp_duckdb,
        job_url="https://example.com/j/2",
        keyword="AI Cybersecurity",
        country="Germany",
        country_code="DE",
        city="Berlin",
    )

    all_df = load_fact_with_dims(tmp_duckdb)
    assert len(all_df) == 2
    assert set(all_df["country_code"]) == {"GB", "DE"}

    filtered = load_fact_with_dims(tmp_duckdb, keyword="Data Engineer")
    assert len(filtered) == 1
    assert filtered.iloc[0]["country_code"] == "GB"
    assert filtered.iloc[0]["salary_currency"] == "USD"


def test_load_fact_with_dims_respects_limit(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    for i in range(5):
        _seed_fact_row(
            tmp_duckdb,
            job_url=f"https://example.com/j/{i}",
            keyword="Data Engineer",
            country="United Kingdom",
            country_code="GB",
        )
    df = load_fact_with_dims(tmp_duckdb, limit=3)
    assert len(df) == 3


def test_load_fact_with_dims_invalid_limit(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(ValueError):
        load_fact_with_dims(tmp_duckdb, limit=0)
