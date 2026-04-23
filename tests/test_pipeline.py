"""Pipeline smoke tests — flow wires tasks in the right order and records runs.

Real JobSpy and LLM calls are mocked out; the only thing under test is the
orchestration logic itself.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from job_research.enricher import EnrichmentSummary
from job_research.scraper import ScrapeRequest, ScrapeResult
from job_research.transform import TransformSummary


@pytest.fixture
def _no_op_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the settings-backed DB at an isolated tmp path."""
    from job_research import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("DATABASE__PATH", str(tmp_path / "pipeline.duckdb"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    yield
    config.get_settings.cache_clear()


def _fake_scrape(run_id: str, requests: list[ScrapeRequest]) -> list[ScrapeResult]:
    return [
        ScrapeResult(
            request=r,
            rows=2,
            per_site_counts={s: 1 for s in r.sites},
            per_site_errors={s: "" for s in r.sites},
        )
        for r in requests
    ]


def _fake_enrich(**_: Any) -> EnrichmentSummary:
    return EnrichmentSummary(attempted=2, succeeded=2, failed=0)


def _fake_transform(**_: Any) -> TransformSummary:
    return TransformSummary(
        dim_location_rows=1,
        dim_salary_rows=1,
        dim_skill_rows=3,
        fact_rows=2,
        bridge_rows=6,
        marts_refreshed=4,
    )


def test_pipeline_success_happy_path(_no_op_db: None) -> None:
    from job_research import pipeline as pl

    with (
        patch.object(pl, "scrape_to_staging", side_effect=_fake_scrape),
        patch.object(pl, "enrich_staging", side_effect=_fake_enrich),
        patch.object(pl, "run_transform", side_effect=_fake_transform),
        patch.object(pl, "build_provider", return_value=MagicMock()),
    ):
        summary = pl.job_research_pipeline(
            keywords=["data engineer"],
            locations=["London"],
            sites=["linkedin"],
        )

    assert summary.status == "success"
    assert summary.scraped_count == 2
    assert summary.enriched_count == 2
    assert summary.transform is not None
    assert summary.transform.fact_rows == 2
    assert summary.finished_at is not None


def test_pipeline_records_failure(_no_op_db: None) -> None:
    from job_research import pipeline as pl

    def _boom(**_: Any) -> EnrichmentSummary:
        raise RuntimeError("llm provider exploded")

    with (
        patch.object(pl, "scrape_to_staging", side_effect=_fake_scrape),
        patch.object(pl, "enrich_staging", side_effect=_boom),
        patch.object(pl, "build_provider", return_value=MagicMock()),
        pytest.raises(RuntimeError, match="llm provider exploded"),
    ):
        pl.job_research_pipeline(keywords=["x"])

    # Run row must be recorded as failed.
    from job_research.database import connect

    with connect() as con:
        rows = con.execute("SELECT status, error_message FROM pipeline_runs").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "failed"
    assert "llm provider exploded" in rows[0][1]


def test_pipeline_builds_one_request_per_keyword_location(
    _no_op_db: None,
) -> None:
    from job_research import pipeline as pl

    captured: dict[str, Any] = {}

    def _capture(run_id: str, requests: list[ScrapeRequest]) -> list[ScrapeResult]:
        captured["requests"] = requests
        return _fake_scrape(run_id, requests)

    with (
        patch.object(pl, "scrape_to_staging", side_effect=_capture),
        patch.object(pl, "enrich_staging", side_effect=_fake_enrich),
        patch.object(pl, "run_transform", side_effect=_fake_transform),
        patch.object(pl, "build_provider", return_value=MagicMock()),
    ):
        pl.job_research_pipeline(
            keywords=["a", "b"],
            locations=["London", "Berlin"],
            sites=["indeed"],
        )

    reqs = captured["requests"]
    assert len(reqs) == 4  # 2 keywords, 2 locations
    pairs = {(r.keyword, r.location) for r in reqs}
    assert pairs == {
        ("a", "London"),
        ("a", "Berlin"),
        ("b", "London"),
        ("b", "Berlin"),
    }
