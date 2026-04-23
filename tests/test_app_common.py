"""Unit tests for pure-Python helpers in job_research.app.common.

These tests never import Streamlit — only the DB helpers and profile CRUD
are exercised.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import duckdb
import pytest

from job_research import constants as C
from job_research.app.common import (
    Profile,
    apply_title_labels_to_judged,
    create_profile_from_name,
    delete_profile,
    delete_title_label,
    ensure_default_profile,
    get_profile,
    get_triage_candidates,
    list_profiles,
    list_title_labels,
    load_fact_with_dims,
    load_mart,
    save_profile,
    save_title_label,
    slugify,
)
from job_research.database import stable_key


# --------------------------------------------------------------------------- #
# Profile CRUD
# --------------------------------------------------------------------------- #
def test_slugify_basic() -> None:
    assert slugify("Bruno — AI/Security/ML") == "bruno-ai-security-ml"
    assert slugify("  Wife Retail  ") == "wife-retail"
    with pytest.raises(ValueError):
        slugify("  ")


def test_list_profiles_empty(tmp_duckdb: duckdb.DuckDBPyConnection) -> None:
    assert list_profiles(tmp_duckdb) == []


def test_save_and_get_profile_roundtrip(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    p = Profile(
        profile_id="bruno-ai-sec",
        name="Bruno AI/Sec",
        description="AI + security + ML",
        keywords=["machine learning engineer", "ai security"],
        locations=["London, UK", "Berlin, Germany"],
        sites=[C.SITE_LINKEDIN, C.SITE_INDEED],
    )
    save_profile(tmp_duckdb, p)
    got = get_profile(tmp_duckdb, "bruno-ai-sec")
    assert got is not None
    assert got.name == "Bruno AI/Sec"
    assert got.keywords == ["machine learning engineer", "ai security"]
    assert got.locations == ["London, UK", "Berlin, Germany"]
    assert got.sites == [C.SITE_LINKEDIN, C.SITE_INDEED]
    assert got.description == "AI + security + ML"


def test_save_profile_strips_and_filters_sites(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    save_profile(
        tmp_duckdb,
        Profile(
            profile_id="p1",
            name="P1",
            keywords=["  ds ", "", "  "],
            locations=["London "],
            sites=[C.SITE_LINKEDIN, "not_a_site"],
        ),
    )
    p = get_profile(tmp_duckdb, "p1")
    assert p is not None
    assert p.keywords == ["ds"]
    assert p.locations == ["London"]
    assert p.sites == [C.SITE_LINKEDIN]


def test_save_profile_rejects_blank_name_and_no_keywords(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(ValueError):
        save_profile(
            tmp_duckdb,
            Profile(profile_id="p", name="   ", keywords=["x"]),
        )
    with pytest.raises(ValueError):
        save_profile(
            tmp_duckdb,
            Profile(profile_id="p", name="P", keywords=["   "]),
        )


def test_save_profile_upsert_overwrites(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    save_profile(
        tmp_duckdb,
        Profile(profile_id="p", name="P", keywords=["old"]),
    )
    save_profile(
        tmp_duckdb,
        Profile(profile_id="p", name="P", keywords=["new"]),
    )
    p = get_profile(tmp_duckdb, "p")
    assert p is not None and p.keywords == ["new"]


def test_delete_profile(tmp_duckdb: duckdb.DuckDBPyConnection) -> None:
    save_profile(
        tmp_duckdb,
        Profile(profile_id="tmp", name="tmp", keywords=["x"]),
    )
    assert get_profile(tmp_duckdb, "tmp") is not None
    delete_profile(tmp_duckdb, "tmp")
    assert get_profile(tmp_duckdb, "tmp") is None


def test_ensure_default_profile_creates_when_empty(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    p = ensure_default_profile(tmp_duckdb)
    assert p.profile_id == C.DEFAULT_PROFILE_ID
    assert p.keywords  # non-empty seed


def test_ensure_default_profile_migrates_legacy_config(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    # Legacy single-row table populated (as pre-profiles DBs had).
    tmp_duckdb.execute(
        "INSERT INTO user_search_config VALUES (?, ?, CURRENT_TIMESTAMP)",
        ["keywords", json.dumps(["legacy-kw"])],
    )
    tmp_duckdb.execute(
        "INSERT INTO user_search_config VALUES (?, ?, CURRENT_TIMESTAMP)",
        ["locations", json.dumps(["Madrid, Spain"])],
    )
    p = ensure_default_profile(tmp_duckdb)
    assert p.keywords == ["legacy-kw"]
    assert p.locations == ["Madrid, Spain"]


def test_ensure_default_profile_idempotent(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    p1 = ensure_default_profile(tmp_duckdb)
    p2 = ensure_default_profile(tmp_duckdb)
    assert p1.profile_id == p2.profile_id
    assert len(list_profiles(tmp_duckdb)) == 1


def test_create_profile_from_name_generates_unique_id(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    a = create_profile_from_name(tmp_duckdb, "Retail Manager")
    b = create_profile_from_name(tmp_duckdb, "Retail Manager")
    assert a.profile_id != b.profile_id
    assert a.profile_id.startswith("retail-manager")


# --------------------------------------------------------------------------- #
# load_mart
# --------------------------------------------------------------------------- #
def test_load_mart_empty_returns_empty_df(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    df = load_mart(tmp_duckdb, "mart_jobs_by_country")
    assert df.empty
    assert "country_code" in df.columns
    assert "job_count" in df.columns


def test_load_mart_with_data_unfiltered(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    tmp_duckdb.execute(
        """
        INSERT INTO mart_jobs_by_country
            (profile_id, country_code, country, search_keyword, job_count)
        VALUES
            ('p1', 'GB', 'United Kingdom', 'Data Engineer', 12),
            ('p2', 'DE', 'Germany',        'Data Engineer',  7)
        """
    )
    df = load_mart(tmp_duckdb, "mart_jobs_by_country")
    assert len(df) == 2
    assert df["job_count"].sum() == 19


def test_load_mart_filters_by_profile(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    tmp_duckdb.execute(
        """
        INSERT INTO mart_jobs_by_country
            (profile_id, country_code, country, search_keyword, job_count)
        VALUES
            ('p1', 'GB', 'United Kingdom', 'Data Engineer', 12),
            ('p2', 'DE', 'Germany',        'Data Engineer',  7)
        """
    )
    df = load_mart(tmp_duckdb, "mart_jobs_by_country", profile_id="p1")
    assert len(df) == 1
    assert df.iloc[0]["country_code"] == "GB"


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
    profile_id: str | None = None,
) -> None:
    """Insert a full fact row with its dim + staging + intermediate rows."""
    job_id = stable_key("linkedin", job_url)
    loc_key = stable_key(city.lower(), country_code)
    sal_key = stable_key(100000.0, 150000.0, "USD", "yearly")
    now = datetime.now(UTC).replace(tzinfo=None)

    con.execute(
        """
        INSERT INTO staging_job_offers
            (id, scraped_at, run_id, profile_id, site, search_keyword, job_url)
        VALUES (?, ?, 'r1', ?, 'linkedin', ?, ?)
        """,
        [job_id, now, profile_id, keyword, job_url],
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
            tech_skills, soft_skills, domain_skills,
            city, country, country_code, work_mode,
            salary_min, salary_max, salary_currency, salary_period
        )
        VALUES (?, ?, 'anthropic', 'claude-haiku-4-5',
                ?, ?, ?, ?, ?, ?, 'remote',
                100000.0, 150000.0, 'USD', 'yearly')
        """,
        [
            job_id,
            now,
            '["python","sql"]',
            '["collaborative"]',
            '["mlops","agile delivery"]',
            city,
            country,
            country_code,
        ],
    )
    con.execute(
        """
        INSERT INTO fact_job_offers (
            job_id, run_id, profile_id, scraped_at, enriched_at, site,
            search_keyword, company, title, job_url, date_posted,
            work_mode, location_key, salary_key
        )
        VALUES (?, 'r1', ?, ?, ?, 'linkedin',
                ?, 'Acme', 'Data Engineer', ?, DATE '2026-04-20',
                'remote', ?, ?)
        """,
        [job_id, profile_id, now, now, keyword, job_url, loc_key, sal_key],
    )


def test_load_fact_with_dims_empty(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    df = load_fact_with_dims(tmp_duckdb)
    assert df.empty
    expected_cols = {
        "job_id",
        "profile_id",
        "site",
        "search_keyword",
        "city",
        "country",
        "salary_min",
        "salary_currency",
        "tech_skills",
        "domain_skills",
    }
    assert expected_cols.issubset(set(df.columns))


def test_load_fact_with_dims_joins_and_filters(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    _seed_fact_row(
        tmp_duckdb,
        job_url="https://example.com/j/1",
        keyword="Data Engineer",
        country="United Kingdom",
        country_code="GB",
        profile_id="bruno",
    )
    _seed_fact_row(
        tmp_duckdb,
        job_url="https://example.com/j/2",
        keyword="Store Development Manager",
        country="Germany",
        country_code="DE",
        city="Berlin",
        profile_id="wife",
    )

    all_df = load_fact_with_dims(tmp_duckdb)
    assert len(all_df) == 2

    by_keyword = load_fact_with_dims(tmp_duckdb, keyword="Data Engineer")
    assert len(by_keyword) == 1
    assert by_keyword.iloc[0]["country_code"] == "GB"

    by_profile = load_fact_with_dims(tmp_duckdb, profile_id="wife")
    assert len(by_profile) == 1
    assert by_profile.iloc[0]["profile_id"] == "wife"


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


# --------------------------------------------------------------------------- #
# Triage label CRUD
# --------------------------------------------------------------------------- #
def test_save_and_list_title_labels(tmp_duckdb: duckdb.DuckDBPyConnection) -> None:
    save_title_label(
        tmp_duckdb,
        profile_id="p1",
        title_norm="data engineer",
        label="accept",
        note="good fit",
    )
    save_title_label(
        tmp_duckdb,
        profile_id="p1",
        title_norm="marketing manager",
        label="reject",
        note=None,
    )
    labels = list_title_labels(tmp_duckdb, "p1")
    assert len(labels) == 2
    by_norm = {r["title_norm"]: r for r in labels}
    assert by_norm["data engineer"]["label"] == "accept"
    assert by_norm["data engineer"]["note"] == "good fit"
    assert by_norm["marketing manager"]["label"] == "reject"
    assert by_norm["marketing manager"]["note"] is None
    # Other profile is isolated
    assert list_title_labels(tmp_duckdb, "p2") == []


def test_save_title_label_upsert_updates_label(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    save_title_label(
        tmp_duckdb, profile_id="p1", title_norm="store manager", label="accept"
    )
    save_title_label(
        tmp_duckdb,
        profile_id="p1",
        title_norm="store manager",
        label="reject",
        note="changed mind",
    )
    labels = list_title_labels(tmp_duckdb, "p1")
    assert len(labels) == 1
    assert labels[0]["label"] == "reject"
    assert labels[0]["note"] == "changed mind"
    # count_seen incremented on conflict
    assert labels[0]["count_seen"] == 2


def test_save_title_label_rejects_invalid_label(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(ValueError, match="accept/reject/unsure"):
        save_title_label(tmp_duckdb, profile_id="p1", title_norm="foo", label="maybe")


def test_delete_title_label(tmp_duckdb: duckdb.DuckDBPyConnection) -> None:
    save_title_label(
        tmp_duckdb, profile_id="p1", title_norm="data engineer", label="accept"
    )
    assert len(list_title_labels(tmp_duckdb, "p1")) == 1
    delete_title_label(tmp_duckdb, profile_id="p1", title_norm="data engineer")
    assert list_title_labels(tmp_duckdb, "p1") == []


def test_apply_labels_overrides_judged_verdict(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    job_url = "https://example.com/j/triage1"
    jid = stable_key("linkedin", job_url)

    # Insert minimal staging row
    tmp_duckdb.execute(
        """
        INSERT INTO staging_job_offers
            (id, scraped_at, run_id, profile_id, site, search_keyword, job_url)
        VALUES (?, ?, 'r1', 'p1', 'linkedin', 'Data Engineer', ?)
        """,
        [jid, now, job_url],
    )
    # Insert judged row with ensemble_verdict = 'review'
    tmp_duckdb.execute(
        """
        INSERT INTO judged_job_offers
            (job_id, profile_id, search_keyword, job_title,
             rule_verdict, ensemble_verdict)
        VALUES (?, 'p1', 'Data Engineer', 'Marketing Manager',
                'reject', 'review')
        """,
        [jid],
    )

    # Save a 'reject' label for the title
    save_title_label(
        tmp_duckdb,
        profile_id="p1",
        title_norm="marketing manager",
        label="reject",
    )

    count = apply_title_labels_to_judged(tmp_duckdb, "p1")
    assert count >= 1

    row = tmp_duckdb.execute(
        "SELECT ensemble_verdict FROM judged_job_offers WHERE job_id = ?", [jid]
    ).fetchone()
    assert row is not None
    assert row[0] == "reject"


def test_get_triage_candidates_returns_unlabelled(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)

    def _seed(url: str, title: str) -> str:
        jid = stable_key("linkedin", url)
        tmp_duckdb.execute(
            """
            INSERT INTO staging_job_offers
                (id, scraped_at, run_id, profile_id, site, search_keyword,
                 job_url, company)
            VALUES (?, ?, 'r1', 'p1', 'linkedin', 'Data Engineer', ?, 'Acme')
            """,
            [jid, now, url],
        )
        tmp_duckdb.execute(
            """
            INSERT INTO judged_job_offers
                (job_id, profile_id, search_keyword, job_title,
                 rule_verdict, ensemble_verdict)
            VALUES (?, 'p1', 'Data Engineer', ?, 'review', 'review')
            """,
            [jid, title],
        )
        return jid

    _seed("https://example.com/j/a", "Data Engineer")
    _seed("https://example.com/j/b", "Marketing Manager")

    # Both titles appear in the default (unlabelled) view
    df = get_triage_candidates(tmp_duckdb, "p1")
    assert len(df) == 2
    assert set(df["title_norm"]) == {"data engineer", "marketing manager"}

    # Label one title
    save_title_label(
        tmp_duckdb, profile_id="p1", title_norm="data engineer", label="accept"
    )

    # Default view (include_decided=False) hides the labelled title
    df2 = get_triage_candidates(tmp_duckdb, "p1")
    assert len(df2) == 1
    assert df2.iloc[0]["title_norm"] == "marketing manager"

    # include_decided=True shows all
    df3 = get_triage_candidates(tmp_duckdb, "p1", include_decided=True)
    assert len(df3) == 2
