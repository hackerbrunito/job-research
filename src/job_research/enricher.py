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

# Graceful degradation: if sentence-transformers isn't installed, every row
# gets score=1.0 and passes through to the LLM exactly as before.
try:
    from job_research.semantic_scorer import score_relevance as _score_relevance

    _SEMANTIC_SCORER_AVAILABLE = True
    _SEMANTIC_SCORE_THRESHOLD = C.SEMANTIC_SCORE_THRESHOLD
except ImportError:
    _SEMANTIC_SCORER_AVAILABLE = False

    def _score_relevance(**_: object) -> float:  # type: ignore[misc]
        return 1.0

    _SEMANTIC_SCORE_THRESHOLD = 0.0

# Graceful degradation: cross-encoder reranker (heavier model, runs after bi-encoder).
try:
    from job_research.cross_encoder_scorer import (
        CROSS_ENCODER_THRESHOLD,
    )
    from job_research.cross_encoder_scorer import (
        cross_encode as _cross_encode,
    )

    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False

    def _cross_encode(**_: object) -> float:  # type: ignore[misc]
        return 1.0

    CROSS_ENCODER_THRESHOLD = 0.0

# Graceful degradation: SetFit few-shot classifier (requires profile training data).
try:
    from job_research.constants import SETFIT_SCORE_THRESHOLD
    from job_research.setfit_classifier import predict as _setfit_predict

    _SETFIT_AVAILABLE = True
except ImportError:
    _SETFIT_AVAILABLE = False

    def _setfit_predict(profile_id: object, texts: list[str]) -> list[float]:  # type: ignore[misc]
        return [1.0] * len(texts)

    SETFIT_SCORE_THRESHOLD = 0.0

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
SELECT s.id, s.title, s.description, s.search_keyword,
       s.profile_id, j.rule_verdict
FROM staging_job_offers AS s
LEFT JOIN int_enriched_job_info AS e ON e.job_id = s.id
LEFT JOIN judged_job_offers AS j ON s.id = j.job_id
WHERE e.job_id IS NULL
  AND (j.rule_verdict IS NULL OR j.rule_verdict != 'reject')
{run_filter}
ORDER BY s.scraped_at
{limit_clause}
"""


def _compute_ensemble(
    rule_verdict: str | None,
    llm_is_relevant: bool,
    biencoder_score: float = 1.0,
    crossencoder_score: float = 0.0,
    setfit_score: float = 1.0,
) -> str:
    """Derive ensemble_verdict from rule filter, all scorer signals, and LLM judge.

    Layered rejection: any single classifier can veto a row.
    A row must pass ALL active classifiers AND the LLM judge to be accepted.
    """
    if rule_verdict == "reject":
        return "reject"
    # Bi-encoder pre-reject: score too low even before consulting the LLM.
    if biencoder_score < _SEMANTIC_SCORE_THRESHOLD:
        return "reject"
    if _CROSS_ENCODER_AVAILABLE and crossencoder_score < CROSS_ENCODER_THRESHOLD:
        return "reject"
    if _SETFIT_AVAILABLE and setfit_score < SETFIT_SCORE_THRESHOLD:
        return "reject"
    if llm_is_relevant:
        return "accept"
    return "reject"


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

        # SetFit: train (or skip if too few labels) for the active profile.
        # We resolve profile_id from the first pending row if not given directly.
        _active_profile_id: str | None = None
        if pending:
            _active_profile_id = pending[0][4]  # profile_id column index
        if _SETFIT_AVAILABLE and _active_profile_id:
            from job_research.app.common import list_title_labels
            from job_research.setfit_classifier import train_for_profile

            try:
                with connect() as label_con:
                    labels = list_title_labels(label_con, _active_profile_id)
                if labels:
                    trained = train_for_profile(_active_profile_id, labels)
                    if trained:
                        log.info(
                            "enricher.setfit.trained", profile_id=_active_profile_id
                        )
            except Exception as exc:
                log.warning("enricher.setfit.train_failed", error=str(exc))

        attempted = 0
        succeeded = 0
        failed = 0
        batch: list[dict[str, Any]] = []

        for row in pending:
            job_id, title, description, search_keyword, profile_id, rule_verdict = row
            attempted += 1

            # --- Bi-encoder pre-filter (runs before LLM to save tokens) ------
            biencoder_score: float = _score_relevance(
                search_keyword=search_keyword or "",
                job_title=title or "",
                job_description=description,
            )

            # --- Cross-encoder + SetFit (only when bi-encoder passes) ---------
            crossencoder_score: float = 0.0
            setfit_score: float = 1.0

            if biencoder_score >= _SEMANTIC_SCORE_THRESHOLD:
                crossencoder_score = _cross_encode(
                    search_keyword=search_keyword or "",
                    job_title=title or "",
                    job_description=description,
                )
                if _SETFIT_AVAILABLE and profile_id:
                    scores = _setfit_predict(profile_id, [title or ""])
                    setfit_score = scores[0]

            if biencoder_score < _SEMANTIC_SCORE_THRESHOLD:
                log.info(
                    "enrich.biencoder_reject",
                    job_id=job_id,
                    score=biencoder_score,
                    threshold=_SEMANTIC_SCORE_THRESHOLD,
                )
                conn.execute(
                    """
                    INSERT INTO judged_job_offers
                        (job_id, profile_id, search_keyword, job_title,
                         rule_verdict, ensemble_verdict, biencoder_score,
                         crossencoder_score, setfit_score,
                         judged_at)
                    VALUES (?, ?, ?, ?, ?, 'reject', ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT (job_id) DO UPDATE SET
                        biencoder_score    = excluded.biencoder_score,
                        crossencoder_score = excluded.crossencoder_score,
                        setfit_score       = excluded.setfit_score,
                        ensemble_verdict   = excluded.ensemble_verdict,
                        judged_at          = excluded.judged_at
                    """,
                    [
                        job_id,
                        profile_id,
                        search_keyword,
                        title,
                        rule_verdict or "accept",
                        biencoder_score,
                        crossencoder_score,
                        setfit_score,
                    ],
                )
                # Count as succeeded (scored and written); no LLM call needed.
                succeeded += 1
                continue
            # ------------------------------------------------------------------

            try:
                enrichment = provider.enrich(
                    title=title or "",
                    description=description or "",
                    search_keyword=search_keyword or "",
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

            # Write LLM verdict to judged_job_offers (upsert).
            ensemble = _compute_ensemble(
                rule_verdict,
                enrichment.is_relevant,
                biencoder_score,
                crossencoder_score,
                setfit_score,
            )
            conn.execute(
                """
                INSERT INTO judged_job_offers
                    (job_id, profile_id, search_keyword, job_title,
                     rule_verdict, llm_is_relevant, llm_confidence,
                     llm_reason, ensemble_verdict, biencoder_score,
                     crossencoder_score, setfit_score,
                     judged_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (job_id) DO UPDATE SET
                    llm_is_relevant    = excluded.llm_is_relevant,
                    llm_confidence     = excluded.llm_confidence,
                    llm_reason         = excluded.llm_reason,
                    ensemble_verdict   = excluded.ensemble_verdict,
                    biencoder_score    = excluded.biencoder_score,
                    crossencoder_score = excluded.crossencoder_score,
                    setfit_score       = excluded.setfit_score,
                    judged_at          = excluded.judged_at
                """,
                [
                    job_id,
                    profile_id,
                    search_keyword,
                    title,
                    rule_verdict or "accept",
                    enrichment.is_relevant,
                    enrichment.relevance_confidence,
                    enrichment.relevance_reason,
                    ensemble,
                    biencoder_score,
                    crossencoder_score,
                    setfit_score,
                ],
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
