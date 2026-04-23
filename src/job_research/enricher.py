"""LLM enrichment orchestration.

Reads staging rows that have not been enriched yet, calls the configured
`LLMProvider`, and writes the structured result into `int_enriched_job_info`.

Failures are logged and counted but do NOT abort the run — we want the
pipeline to make forward progress on the rows that can be enriched.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import duckdb
import pandas as pd
from pydantic import ValidationError

from job_research import constants as C
from job_research.config import get_settings
from job_research.database import connect, insert_dataframe
from job_research.llm_providers import LLMProvider, build_provider
from job_research.logging_setup import get_logger
from job_research.schemas import JobEnrichment

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EnrichmentSummary:
    """Result of an `enrich_staging` run."""

    attempted: int
    succeeded: int
    failed: int


# --------------------------------------------------------------------------- #
# Row shaping
# --------------------------------------------------------------------------- #
def _enrichment_to_row(
    *,
    job_id: str,
    enrichment: JobEnrichment,
    provider_name: str,
    model_name: str,
) -> dict[str, Any]:
    """Flatten a `JobEnrichment` into a row matching `int_enriched_job_info`."""
    raw = enrichment.model_dump(mode="json")
    return {
        "job_id": job_id,
        "enriched_at": datetime.now(UTC).replace(tzinfo=None),
        "llm_provider": provider_name,
        "llm_model": model_name,
        "tech_skills": enrichment.tech_skills,
        "soft_skills": enrichment.soft_skills,
        "domain_skills": enrichment.domain_skills,
        "city": enrichment.location.city,
        "country": enrichment.location.country,
        "country_code": enrichment.location.country_code,
        "work_mode": enrichment.work_mode,
        "salary_min": enrichment.salary.min_amount,
        "salary_max": enrichment.salary.max_amount,
        "salary_currency": enrichment.salary.currency,
        "salary_period": enrichment.salary.period,
        "raw_response": raw,
    }


def _flush(con: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> int:
    """Insert a batch of enrichment rows. Returns the count inserted."""
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    # DuckDB JSON columns accept Python lists/dicts natively when inserted via
    # DataFrame registration, but pyarrow conversion prefers strings. We let
    # the default pandas->duckdb mapping handle it; DuckDB will coerce lists
    # of strings and dicts into JSON when the target column is JSON.
    return insert_dataframe(con, df, table="int_enriched_job_info")


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
_STAGING_SELECT = """
SELECT s.id, s.title, s.description
FROM staging_job_offers AS s
LEFT JOIN int_enriched_job_info AS e ON e.job_id = s.id
WHERE e.job_id IS NULL
{run_filter}
ORDER BY s.scraped_at
{limit_clause}
"""


def enrich_staging(
    run_id: str | None = None,
    *,
    provider: LLMProvider | None = None,
    limit: int | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> EnrichmentSummary:
    """Enrich every `staging_job_offers` row not yet in `int_enriched_job_info`.

    Args:
        run_id: if given, only rows from this pipeline run are considered.
        provider: optional pre-built provider; if omitted, one is built from
            application settings.
        limit: hard cap on number of rows to attempt (useful for dev).
        con: an existing DuckDB connection (used by tests / longer
            pipelines). If omitted a new one is opened and closed.

    Returns:
        EnrichmentSummary with attempted / succeeded / failed counts.
    """
    settings = get_settings()
    if provider is None:
        provider = build_provider(settings.llm, settings)

    params: list[Any] = []
    run_filter = ""
    if run_id is not None:
        run_filter = "AND s.run_id = ?"
        params.append(run_id)

    limit_clause = ""
    if limit is not None:
        if limit <= 0:
            return EnrichmentSummary(0, 0, 0)
        limit_clause = f"LIMIT {int(limit)}"

    sql = _STAGING_SELECT.format(run_filter=run_filter, limit_clause=limit_clause)

    ctx = nullcontext(con) if con is not None else connect()
    with ctx as conn:
        pending = conn.execute(sql, params).fetchall()
        log.info(
            "enrich.start",
            run_id=run_id,
            pending=len(pending),
            provider=provider.provider_name,
            model=provider.model_name,
        )

        attempted = 0
        succeeded = 0
        failed = 0
        batch: list[dict[str, Any]] = []

        for row in pending:
            job_id, title, description = row
            attempted += 1
            try:
                enrichment = provider.enrich(
                    title=title or "", description=description or ""
                )
            except ValidationError as exc:
                failed += 1
                log.warning(
                    "enrich.validation_error",
                    job_id=job_id,
                    error=str(exc),
                )
                continue
            except Exception as exc:
                failed += 1
                log.warning(
                    "enrich.provider_error",
                    job_id=job_id,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )
                continue

            batch.append(
                _enrichment_to_row(
                    job_id=job_id,
                    enrichment=enrichment,
                    provider_name=provider.provider_name,
                    model_name=provider.model_name,
                )
            )
            succeeded += 1

            if len(batch) >= C.ENRICH_BATCH_SIZE:
                _flush(conn, batch)
                batch.clear()

        if batch:
            _flush(conn, batch)

    summary = EnrichmentSummary(attempted=attempted, succeeded=succeeded, failed=failed)
    log.info("enrich.done", **summary.__dict__)
    return summary


__all__ = ["EnrichmentSummary", "enrich_staging"]
