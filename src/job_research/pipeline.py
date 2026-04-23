"""Prefect 3 flow: scrape → enrich → transform.

Run:
    uv run python -m job_research.pipeline              # default search
    uv run python -m job_research.pipeline --help

This wires the three stage modules together, records the run in
`pipeline_runs`, and returns a structured summary. The flow is the only
public orchestration surface; individual stage modules remain
library-agnostic.
"""

from __future__ import annotations

import argparse
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from job_research import constants as C
from job_research.config import Settings, get_settings
from job_research.database import (
    connect,
    init_schema,
    record_run_finish,
    record_run_start,
)
from job_research.enricher import EnrichmentSummary, enrich_staging
from job_research.llm_providers import build_provider
from job_research.logging_setup import configure_logging
from job_research.scraper import ScrapeRequest, ScrapeResult, scrape_to_staging
from job_research.transform import TransformSummary, run_transform


# --------------------------------------------------------------------------- #
# Summary dataclass
# --------------------------------------------------------------------------- #
@dataclass
class PipelineSummary:
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "running"
    scrape_results: list[ScrapeResult] = field(default_factory=list)
    enrichment: EnrichmentSummary | None = None
    transform: TransformSummary | None = None
    error: str | None = None

    @property
    def scraped_count(self) -> int:
        return sum(r.rows for r in self.scrape_results)

    @property
    def enriched_count(self) -> int:
        return self.enrichment.succeeded if self.enrichment else 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["finished_at"] = self.finished_at.isoformat() if self.finished_at else None
        d["scraped_count"] = self.scraped_count
        d["enriched_count"] = self.enriched_count
        return d


# --------------------------------------------------------------------------- #
# Tasks — thin wrappers that delegate to the stage modules
# --------------------------------------------------------------------------- #
@task(
    name="scrape",
    retries=C.FLOW_RETRY_COUNT,
    retry_delay_seconds=C.FLOW_RETRY_DELAY_SECONDS,
    log_prints=False,
    cache_policy=NO_CACHE,
)
def scrape_task(run_id: str, requests: list[ScrapeRequest]) -> list[ScrapeResult]:
    logger = get_run_logger()
    logger.info(f"scrape.start run_id={run_id} requests={len(requests)}")
    results = scrape_to_staging(run_id=run_id, requests=requests)
    logger.info(
        f"scrape.end run_id={run_id} "
        f"rows={sum(r.rows for r in results)} "
        f"errors={sum(1 for r in results for e in r.per_site_errors.values() if e)}"
    )
    return results


@task(
    name="enrich",
    retries=C.FLOW_RETRY_COUNT,
    retry_delay_seconds=C.FLOW_RETRY_DELAY_SECONDS,
    log_prints=False,
    cache_policy=NO_CACHE,
)
def enrich_task(
    run_id: str,
    limit: int | None = None,
    *,
    settings: Settings,
) -> EnrichmentSummary:
    logger = get_run_logger()
    provider = build_provider(settings.llm, settings)
    logger.info(
        f"enrich.start run_id={run_id} "
        f"provider={settings.llm.provider} model={settings.llm.model}"
    )
    summary = enrich_staging(run_id=run_id, provider=provider, limit=limit)
    logger.info(
        f"enrich.end attempted={summary.attempted} "
        f"succeeded={summary.succeeded} failed={summary.failed}"
    )
    return summary


@task(
    name="transform",
    retries=C.FLOW_RETRY_COUNT,
    retry_delay_seconds=C.FLOW_RETRY_DELAY_SECONDS,
    log_prints=False,
    cache_policy=NO_CACHE,
)
def transform_task() -> TransformSummary:
    logger = get_run_logger()
    logger.info("transform.start")
    summary = run_transform()
    logger.info(
        f"transform.end fact_rows={summary.fact_rows} "
        f"bridge_rows={summary.bridge_rows} "
        f"marts_refreshed={summary.marts_refreshed}"
    )
    return summary


# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #
@flow(
    name="job-research-pipeline",
    description="Scrape → enrich → transform job listings.",
    log_prints=False,
)
def job_research_pipeline(
    keywords: list[str],
    locations: list[str] | None = None,
    sites: list[str] | None = None,
    *,
    enrich_limit: int | None = None,
    settings: Settings | None = None,
) -> PipelineSummary:
    """End-to-end pipeline. One run = one row in `pipeline_runs`.

    Args:
        keywords: Search terms to scrape for. One scrape call per (keyword x location).
        locations: Locations to restrict search to. If None, scrape without location.
        sites: Job boards to scrape. Defaults to DEFAULT_SITES.
        enrich_limit: Optional cap on rows to enrich (for dev/debug).
        settings: Explicit Settings override (e.g. for a Streamlit session
            that wants a non-default LLM provider without mutating process-
            wide state). Defaults to `get_settings()`.
    """
    logger = get_run_logger()
    if settings is None:
        settings = get_settings()
    run_id = uuid.uuid4().hex
    started = datetime.now(UTC).replace(tzinfo=None)

    site_tuple = tuple(sites) if sites else C.DEFAULT_SITES
    locs = locations or [None]  # type: ignore[list-item]
    requests = [
        ScrapeRequest(keyword=k, location=loc, sites=site_tuple)
        for k in keywords
        for loc in locs
    ]

    summary = PipelineSummary(run_id=run_id, started_at=started)

    # Ensure schema exists and record run start before any work.
    init_schema()
    with connect() as con:
        record_run_start(
            con,
            run_id,
            keywords=keywords,
            locations=[loc for loc in locs if loc is not None],
            sites=list(site_tuple),
        )

    try:
        summary.scrape_results = scrape_task(run_id, requests)
        summary.enrichment = enrich_task(run_id, enrich_limit, settings=settings)
        summary.transform = transform_task()
        summary.status = "success"
    except Exception as exc:
        summary.status = "failed"
        summary.error = f"{type(exc).__name__}: {exc}"
        logger.exception("pipeline.failed")
        # fall through to record_run_finish, then re-raise
        with connect() as con:
            record_run_finish(
                con,
                run_id,
                status="failed",
                scraped_count=summary.scraped_count,
                enriched_count=summary.enriched_count,
                error_message=summary.error,
            )
        summary.finished_at = datetime.now(UTC).replace(tzinfo=None)
        raise

    summary.finished_at = datetime.now(UTC).replace(tzinfo=None)
    with connect() as con:
        record_run_finish(
            con,
            run_id,
            status=summary.status,
            scraped_count=summary.scraped_count,
            enriched_count=summary.enriched_count,
        )
    return summary


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the job-research pipeline.")
    parser.add_argument(
        "--keyword",
        "-k",
        action="append",
        required=True,
        help="Search keyword (repeatable).",
    )
    parser.add_argument(
        "--location",
        "-l",
        action="append",
        default=None,
        help="Location filter (repeatable). Omit for no location filter.",
    )
    parser.add_argument(
        "--site",
        "-s",
        action="append",
        default=None,
        choices=list(C.ALL_SITES),
        help=f"Site to scrape (repeatable). Default: {','.join(C.DEFAULT_SITES)}.",
    )
    parser.add_argument(
        "--enrich-limit",
        type=int,
        default=None,
        help="Cap on rows to enrich (for dev/debug).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override log level (DEBUG/INFO/WARNING/ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(level=args.log_level or settings.log_level)

    summary = job_research_pipeline(
        keywords=args.keyword,
        locations=args.location,
        sites=args.site,
        enrich_limit=args.enrich_limit,
    )
    print(
        f"[{summary.status}] run_id={summary.run_id} "
        f"scraped={summary.scraped_count} enriched={summary.enriched_count}"
    )
    return 0 if summary.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
