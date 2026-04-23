"""Tests for the corrective loop module."""

from __future__ import annotations

import duckdb
import pytest
from pydantic import SecretStr

from job_research.config import LLMConfig, Settings
from job_research.corrective_loop import (
    compute_acceptance_rates,
    propose_alternatives,
    run_corrective_pass,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _insert_staging_and_judged(
    con: duckdb.DuckDBPyConnection,
    *,
    job_id: str,
    title: str,
    keyword: str,
    run_id: str = "r1",
    profile_id: str = "p1",
    ensemble_verdict: str = "accept",
) -> None:
    """Insert a minimal staging row + judged row for test setup."""
    con.execute(
        """
        INSERT OR IGNORE INTO staging_job_offers
            (id, scraped_at, run_id, profile_id, site,
             search_keyword, search_location, job_url, title, description)
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, 'linkedin', ?, NULL, ?, ?, '')
        """,
        [job_id, run_id, profile_id, keyword, f"https://example.com/{job_id}", title],
    )
    con.execute(
        """
        INSERT OR IGNORE INTO judged_job_offers
            (job_id, profile_id, search_keyword, job_title,
             rule_verdict, ensemble_verdict, judged_at)
        VALUES (?, ?, ?, ?, 'accept', ?, CURRENT_TIMESTAMP)
        """,
        [job_id, profile_id, keyword, title, ensemble_verdict],
    )


def _make_settings(provider: str = "anthropic") -> Settings:
    """Build a minimal Settings for tests (no real API calls)."""
    return Settings(
        anthropic_api_key=SecretStr("test-key"),
        llm=LLMConfig(provider=provider, model="claude-haiku-4-5-20251001"),
    )


# --------------------------------------------------------------------------- #
# compute_acceptance_rates
# --------------------------------------------------------------------------- #
def test_compute_acceptance_rates_empty(tmp_duckdb: duckdb.DuckDBPyConnection) -> None:
    rates = compute_acceptance_rates("run-x", "p1", tmp_duckdb)
    assert rates == {}


def test_compute_acceptance_rates_partial(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """Insert 2 accept + 3 reject for keyword 'Store Development Manager'."""
    kw = "Store Development Manager"
    for i, verdict in enumerate(["accept", "accept", "reject", "reject", "reject"]):
        _insert_staging_and_judged(
            tmp_duckdb,
            job_id=f"job-{i}",
            title=f"Title {i}",
            keyword=kw,
            run_id="r1",
            profile_id="p1",
            ensemble_verdict=verdict,
        )

    rates = compute_acceptance_rates("r1", "p1", tmp_duckdb)
    assert kw in rates
    assert abs(rates[kw] - 0.4) < 0.01


def test_compute_acceptance_rates_filters_by_profile(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """Rows for a different profile_id must not be included."""
    kw = "Data Engineer"
    _insert_staging_and_judged(
        tmp_duckdb,
        job_id="j1",
        title="Data Engineer",
        keyword=kw,
        run_id="r1",
        profile_id="p1",
        ensemble_verdict="accept",
    )
    # Same run_id, different profile
    con = tmp_duckdb
    con.execute(
        """
        INSERT OR IGNORE INTO staging_job_offers
            (id, scraped_at, run_id, profile_id, site,
             search_keyword, search_location, job_url, title, description)
        VALUES ('j2', CURRENT_TIMESTAMP, 'r1', 'p2', 'linkedin', ?, NULL,
                'https://example.com/j2', 'Unrelated', '')
        """,
        [kw],
    )
    con.execute(
        """
        INSERT OR IGNORE INTO judged_job_offers
            (job_id, profile_id, search_keyword, job_title,
             rule_verdict, ensemble_verdict, judged_at)
        VALUES ('j2', 'p2', ?, 'Unrelated', 'accept', 'reject', CURRENT_TIMESTAMP)
        """,
        [kw],
    )

    rates_p1 = compute_acceptance_rates("r1", "p1", tmp_duckdb)
    assert abs(rates_p1[kw] - 1.0) < 0.01

    rates_p2 = compute_acceptance_rates("r1", "p2", tmp_duckdb)
    assert abs(rates_p2[kw] - 0.0) < 0.01


def test_compute_acceptance_rates_no_profile_filter(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """profile_id=None aggregates across all profiles."""
    kw = "ML Engineer"
    for i, (pid, verdict) in enumerate(
        [("p1", "accept"), ("p2", "accept"), ("p3", "reject")]
    ):
        _insert_staging_and_judged(
            tmp_duckdb,
            job_id=f"ml-{i}",
            title="ML Engineer",
            keyword=kw,
            run_id="r2",
            profile_id=pid,
            ensemble_verdict=verdict,
        )

    rates = compute_acceptance_rates("r2", None, tmp_duckdb)
    assert kw in rates
    assert abs(rates[kw] - 2 / 3) < 0.01


# --------------------------------------------------------------------------- #
# propose_alternatives
# --------------------------------------------------------------------------- #
def test_propose_alternatives_returns_list(mocker: pytest.MonkeyPatch) -> None:
    """Mock Anthropic SDK; verify parsing works."""
    mock_client = mocker.MagicMock()
    mock_client.messages.create.return_value.content = [
        mocker.MagicMock(
            text='["Store Openings Manager", "Retail Real Estate Manager"]'
        )
    ]
    mocker.patch("anthropic.Anthropic", return_value=mock_client)

    settings = _make_settings(provider="anthropic")
    result = propose_alternatives(
        "Store Development Manager",
        ["Store Openings Director"],
        ["Lawn Operative"],
        settings=settings,
    )
    assert result == ["Store Openings Manager", "Retail Real Estate Manager"]


def test_propose_alternatives_gracefully_handles_bad_json(
    mocker: pytest.MonkeyPatch,
) -> None:
    """Non-JSON LLM response must return []."""
    mock_client = mocker.MagicMock()
    mock_client.messages.create.return_value.content = [
        mocker.MagicMock(text="I cannot suggest alternatives.")
    ]
    mocker.patch("anthropic.Anthropic", return_value=mock_client)

    settings = _make_settings(provider="anthropic")
    result = propose_alternatives("kw", [], [], settings=settings)
    assert result == []


def test_propose_alternatives_handles_sdk_exception(
    mocker: pytest.MonkeyPatch,
) -> None:
    """Any exception from the SDK must return []."""
    mocker.patch("anthropic.Anthropic", side_effect=RuntimeError("network error"))

    settings = _make_settings(provider="anthropic")
    result = propose_alternatives(
        "kw", ["Good Title"], ["Bad Title"], settings=settings
    )
    assert result == []


def test_propose_alternatives_non_list_json(mocker: pytest.MonkeyPatch) -> None:
    """JSON object (not array) must return []."""
    mock_client = mocker.MagicMock()
    mock_client.messages.create.return_value.content = [
        mocker.MagicMock(text='{"keyword": "something"}')
    ]
    mocker.patch("anthropic.Anthropic", return_value=mock_client)

    settings = _make_settings(provider="anthropic")
    result = propose_alternatives("kw", [], [], settings=settings)
    assert result == []


# --------------------------------------------------------------------------- #
# run_corrective_pass
# --------------------------------------------------------------------------- #
def test_run_corrective_pass_no_trigger_when_rate_ok(
    tmp_duckdb: duckdb.DuckDBPyConnection,
    mocker: pytest.MonkeyPatch,
) -> None:
    """High acceptance rate → no corrective pass fired; triggered=False."""
    kw = "Senior Data Engineer"
    for i in range(5):
        _insert_staging_and_judged(
            tmp_duckdb,
            job_id=f"ok-{i}",
            title=kw,
            keyword=kw,
            run_id="r1",
            profile_id="p1",
            ensemble_verdict="accept",
        )

    # Spy to ensure propose_alternatives is never called.
    spy = mocker.patch(
        "job_research.corrective_loop.propose_alternatives", return_value=[]
    )

    settings = _make_settings()
    results = run_corrective_pass("r1", "p1", ("linkedin",), settings, tmp_duckdb)

    assert all(not r.triggered for r in results)
    spy.assert_not_called()


def test_run_corrective_pass_no_verdicts_returns_empty(
    tmp_duckdb: duckdb.DuckDBPyConnection,
) -> None:
    """No judged rows → returns []."""
    settings = _make_settings()
    results = run_corrective_pass("no-run", "p1", ("linkedin",), settings, tmp_duckdb)
    assert results == []


def test_run_corrective_pass_triggers_on_low_rate(
    tmp_duckdb: duckdb.DuckDBPyConnection,
    mocker: pytest.MonkeyPatch,
) -> None:
    """Low acceptance rate → triggered=True, alternatives populated."""
    kw = "Store Development Manager"
    # 1 accept, 9 reject → 10% acceptance rate (below 30% threshold)
    for i, verdict in enumerate(["accept"] + ["reject"] * 9):
        _insert_staging_and_judged(
            tmp_duckdb,
            job_id=f"low-{i}",
            title=f"Title {i}",
            keyword=kw,
            run_id="r1",
            profile_id="p1",
            ensemble_verdict=verdict,
        )

    # Mock the LLM proposal and downstream scrape+enrich.
    mocker.patch(
        "job_research.corrective_loop.propose_alternatives",
        return_value=["Retail Expansion Manager", "Store Opening Lead"],
    )
    mocker.patch(
        "job_research.corrective_loop.scrape_to_staging",
        return_value=[],
    )
    mock_enrich = mocker.MagicMock()
    mock_enrich.succeeded = 0
    mocker.patch(
        "job_research.corrective_loop.enrich_staging",
        return_value=mock_enrich,
    )

    settings = _make_settings()
    results = run_corrective_pass("r1", "p1", ("linkedin",), settings, tmp_duckdb)

    triggered = [r for r in results if r.triggered]
    assert len(triggered) == 1
    assert triggered[0].original_keyword == kw
    assert triggered[0].alternative_keywords == [
        "Retail Expansion Manager",
        "Store Opening Lead",
    ]
    assert triggered[0].acceptance_rate < 0.30


def test_run_corrective_pass_no_alternatives_no_scrape(
    tmp_duckdb: duckdb.DuckDBPyConnection,
    mocker: pytest.MonkeyPatch,
) -> None:
    """If LLM returns no alternatives, triggered=True but no scrape happens."""
    kw = "Obscure Role"
    for i in range(5):
        _insert_staging_and_judged(
            tmp_duckdb,
            job_id=f"obs-{i}",
            title="Unrelated Title",
            keyword=kw,
            run_id="r1",
            profile_id="p1",
            ensemble_verdict="reject",
        )

    mocker.patch(
        "job_research.corrective_loop.propose_alternatives",
        return_value=[],
    )
    scrape_spy = mocker.patch("job_research.corrective_loop.scrape_to_staging")

    settings = _make_settings()
    results = run_corrective_pass("r1", "p1", ("linkedin",), settings, tmp_duckdb)

    triggered = [r for r in results if r.triggered]
    assert len(triggered) == 1
    assert triggered[0].alternative_keywords == []
    assert triggered[0].new_rows_scraped == 0
    scrape_spy.assert_not_called()
