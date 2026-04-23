"""Tests for job_research.scraper.

All jobspy network calls are mocked via pytest-mock.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import duckdb
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from job_research import scraper
from job_research.constants import SITE_INDEED, SITE_LINKEDIN
from job_research.database import job_id
from job_research.scraper import ScrapeRequest, scrape_to_staging


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame with the JobSpy column set, filling missing cols with None."""
    base_cols = [
        "site",
        "job_url",
        "job_url_direct",
        "title",
        "company",
        "company_url",
        "location",
        "date_posted",
        "job_type",
        "min_amount",
        "max_amount",
        "currency",
        "interval",
        "is_remote",
        "description",
        "company_industry",
        "salary_source",
    ]
    normalized = []
    for r in rows:
        merged = {c: r.get(c) for c in base_cols}
        normalized.append(merged)
    return pd.DataFrame(normalized, columns=base_cols)


@pytest.fixture(autouse=True)
def _no_sleep(mocker: MockerFixture) -> None:
    """Avoid the polite sleep between calls in tests."""
    mocker.patch.object(scraper, "_polite_sleep", lambda _s: None)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_happy_path_two_sites_inserts_rows(
    tmp_duckdb: duckdb.DuckDBPyConnection, mocker: MockerFixture
) -> None:
    linkedin_df = _make_df(
        [
            {
                "site": SITE_LINKEDIN,
                "job_url": "https://linkedin.com/jobs/1",
                "title": "Data Engineer",
                "company": "Acme",
                "location": "Remote",
                "date_posted": date(2026, 4, 20),
                "job_type": "fulltime",
                "is_remote": True,
                "description": "desc li",
            }
        ]
    )
    indeed_df = _make_df(
        [
            {
                "site": SITE_INDEED,
                "job_url": "https://indeed.com/jobs/2",
                "title": "ML Engineer",
                "company": "Foo Corp",
                "location": "NYC",
                "date_posted": date(2026, 4, 21),
                "job_type": "fulltime",
                "is_remote": False,
                "description": "desc in",
            }
        ]
    )

    mock_scrape = mocker.patch.object(scraper, "scrape_jobs")
    mock_scrape.side_effect = [linkedin_df, indeed_df]

    req = ScrapeRequest(
        keyword="data engineer",
        location="NYC",
        sites=(SITE_LINKEDIN, SITE_INDEED),
    )
    results = scrape_to_staging("run-1", [req], con=tmp_duckdb)

    assert len(results) == 1
    res = results[0]
    assert res.rows == 2
    assert res.per_site_counts == {SITE_LINKEDIN: 1, SITE_INDEED: 1}
    assert res.per_site_errors == {SITE_LINKEDIN: "", SITE_INDEED: ""}

    rows = tmp_duckdb.execute(
        "SELECT run_id, site, search_keyword, search_location, job_url, title "
        "FROM staging_job_offers ORDER BY site"
    ).fetchall()
    assert len(rows) == 2
    sites = {r[1] for r in rows}
    assert sites == {SITE_LINKEDIN, SITE_INDEED}
    for run_id_, _site, kw, loc, _url, _title in rows:
        assert run_id_ == "run-1"
        assert kw == "data engineer"
        assert loc == "NYC"


def test_one_site_fails_other_succeeds(
    tmp_duckdb: duckdb.DuckDBPyConnection, mocker: MockerFixture
) -> None:
    indeed_df = _make_df(
        [
            {
                "site": SITE_INDEED,
                "job_url": "https://indeed.com/jobs/ok",
                "title": "Engineer",
                "company": "Co",
            }
        ]
    )

    def side_effect(*_args: Any, **kwargs: Any) -> pd.DataFrame:
        sites = kwargs.get("site_name") or []
        if SITE_LINKEDIN in sites:
            raise RuntimeError("HTTP 429 rate limited")
        return indeed_df

    mocker.patch.object(scraper, "scrape_jobs", side_effect=side_effect)

    req = ScrapeRequest(
        keyword="python", location=None, sites=(SITE_LINKEDIN, SITE_INDEED)
    )
    results = scrape_to_staging("run-2", [req], con=tmp_duckdb)

    res = results[0]
    assert res.rows == 1
    assert res.per_site_counts[SITE_INDEED] == 1
    assert res.per_site_counts[SITE_LINKEDIN] == 0
    assert "429" in res.per_site_errors[SITE_LINKEDIN]
    assert res.per_site_errors[SITE_INDEED] == ""

    count = tmp_duckdb.execute(
        "SELECT count(*) FROM staging_job_offers WHERE run_id = 'run-2'"
    ).fetchone()
    assert count is not None
    assert count[0] == 1


def test_dedup_same_job_url_in_same_site(
    tmp_duckdb: duckdb.DuckDBPyConnection, mocker: MockerFixture
) -> None:
    """Duplicate (site, job_url) across calls only inserts one row."""
    df_a = _make_df(
        [
            {
                "site": SITE_INDEED,
                "job_url": "https://indeed.com/dup",
                "title": "Eng",
            },
            {
                "site": SITE_INDEED,
                "job_url": "https://indeed.com/dup",  # duplicate within same scrape
                "title": "Eng",
            },
            {
                "site": SITE_INDEED,
                "job_url": "https://indeed.com/unique",
                "title": "Other",
            },
        ]
    )
    # Second request returns one row with the same (site, job_url) as before.
    df_b = _make_df(
        [
            {
                "site": SITE_INDEED,
                "job_url": "https://indeed.com/dup",
                "title": "Eng",
            }
        ]
    )

    mocker.patch.object(scraper, "scrape_jobs", side_effect=[df_a, df_b])

    req_a = ScrapeRequest(keyword="a", sites=(SITE_INDEED,))
    req_b = ScrapeRequest(keyword="b", sites=(SITE_INDEED,))
    results = scrape_to_staging("run-3", [req_a, req_b], con=tmp_duckdb)

    # First request: 2 unique rows (one of the three was a duplicate in-batch).
    assert results[0].rows == 2
    # Second request: the single returned row collides with existing PK => 0 insert.
    assert results[1].rows == 0

    total = tmp_duckdb.execute("SELECT count(*) FROM staging_job_offers").fetchone()
    assert total is not None
    assert total[0] == 2

    expected_dup_id = job_id(SITE_INDEED, "https://indeed.com/dup")
    got = tmp_duckdb.execute(
        "SELECT count(*) FROM staging_job_offers WHERE id = ?", [expected_dup_id]
    ).fetchone()
    assert got is not None
    assert got[0] == 1


def test_scrape_retries_on_transient_error(
    tmp_duckdb: duckdb.DuckDBPyConnection, mocker: MockerFixture
) -> None:
    """Tenacity retries a ConnectionError, then succeeds on second attempt."""
    # Avoid actual backoff delay.
    mocker.patch("tenacity.nap.time.sleep", return_value=None)

    empty_df = _make_df([])
    mock_scrape = mocker.patch.object(
        scraper,
        "scrape_jobs",
        side_effect=[ConnectionError("boom"), empty_df],
    )

    req = ScrapeRequest(keyword="python", sites=(SITE_INDEED,))
    results = scrape_to_staging("run-retry", [req], con=tmp_duckdb)

    # Two attempts: first raises, second succeeds with empty df.
    assert mock_scrape.call_count == 2
    assert results[0].per_site_counts[SITE_INDEED] == 0
    # Empty string (not an error) because the retry succeeded.
    assert results[0].per_site_errors[SITE_INDEED] == ""


def test_scrape_does_not_retry_on_non_transient(
    tmp_duckdb: duckdb.DuckDBPyConnection, mocker: MockerFixture
) -> None:
    """ValueError is not in _TRANSIENT_ERRORS -> no retry, captured per-site."""
    mocker.patch("tenacity.nap.time.sleep", return_value=None)

    mock_scrape = mocker.patch.object(
        scraper,
        "scrape_jobs",
        side_effect=ValueError("bad input"),
    )

    req = ScrapeRequest(keyword="python", sites=(SITE_INDEED,))
    results = scrape_to_staging("run-no-retry", [req], con=tmp_duckdb)

    assert mock_scrape.call_count == 1
    assert "ValueError" in results[0].per_site_errors[SITE_INDEED]
    assert results[0].per_site_counts[SITE_INDEED] == 0


def test_missing_job_url_rows_filtered(
    tmp_duckdb: duckdb.DuckDBPyConnection, mocker: MockerFixture
) -> None:
    df = _make_df(
        [
            {"site": SITE_INDEED, "job_url": None, "title": "Skip 1"},
            {"site": SITE_INDEED, "job_url": "", "title": "Skip 2"},
            {"site": SITE_INDEED, "job_url": "   ", "title": "Skip 3"},
            {
                "site": SITE_INDEED,
                "job_url": "https://indeed.com/keep",
                "title": "Keep",
            },
        ]
    )

    mocker.patch.object(scraper, "scrape_jobs", return_value=df)

    req = ScrapeRequest(keyword="x", sites=(SITE_INDEED,))
    results = scrape_to_staging("run-4", [req], con=tmp_duckdb)

    assert results[0].rows == 1
    rows = tmp_duckdb.execute(
        "SELECT job_url FROM staging_job_offers WHERE run_id = 'run-4'"
    ).fetchall()
    assert rows == [("https://indeed.com/keep",)]


# --------------------------------------------------------------------------- #
# country_indeed parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("London, UK", "uk"),
        ("London, United Kingdom", "uk"),
        ("Berlin, Germany", "germany"),
        ("Spain", "spain"),
        ("Hong Kong", "hongkong"),
        ("San Francisco, USA", "usa"),
        ("", None),
        (None, None),
        ("Atlantis", None),
        ("Some City, Nowhere", None),
    ],
)
def test_extract_country_indeed(location: str | None, expected: str | None) -> None:
    assert scraper._extract_country_indeed(location) == expected


def test_indeed_scrape_passes_country_indeed(mocker: MockerFixture) -> None:
    """When site=indeed, country_indeed is inferred from location and forwarded."""
    mock = mocker.patch.object(scraper, "scrape_jobs", return_value=pd.DataFrame())
    scraper._scrape_one_site(
        site=SITE_INDEED,
        keyword="x",
        location="Berlin, Germany",
        cfg=scraper.ScrapingConfig(),
    )
    kwargs = mock.call_args.kwargs
    assert kwargs.get("country_indeed") == "germany"


def test_linkedin_scrape_does_not_pass_country_indeed(mocker: MockerFixture) -> None:
    """country_indeed must only be sent to the indeed backend."""
    mock = mocker.patch.object(scraper, "scrape_jobs", return_value=pd.DataFrame())
    scraper._scrape_one_site(
        site=SITE_LINKEDIN,
        keyword="x",
        location="Berlin, Germany",
        cfg=scraper.ScrapingConfig(),
    )
    assert "country_indeed" not in mock.call_args.kwargs


def test_indeed_scrape_omits_country_indeed_when_unresolved(
    mocker: MockerFixture,
) -> None:
    """Unresolved country -> omit the kwarg so jobspy's own default kicks in."""
    mock = mocker.patch.object(scraper, "scrape_jobs", return_value=pd.DataFrame())
    scraper._scrape_one_site(
        site=SITE_INDEED,
        keyword="x",
        location="Atlantis",
        cfg=scraper.ScrapingConfig(),
    )
    assert "country_indeed" not in mock.call_args.kwargs
