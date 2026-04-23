"""Corrective loop for low-relevance scrape results.

When the acceptance rate for a (profile_id, keyword) pair drops below
CORRECTIVE_ACCEPTANCE_THRESHOLD, this module asks the configured LLM to
propose up to CORRECTIVE_MAX_ALTERNATIVES alternative search phrases,
then re-scrapes with those phrases.

The corrective pass respects CORRECTIVE_MAX_PASSES — it fires at most
once per pipeline run to bound API cost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import duckdb

from job_research import constants as C
from job_research.config import Settings
from job_research.database import connect
from job_research.enricher import enrich_staging
from job_research.llm_providers import build_provider
from job_research.logging_setup import get_logger
from job_research.scraper import ScrapeRequest, scrape_to_staging

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Public dataclass
# --------------------------------------------------------------------------- #
@dataclass
class CorrectiveResult:
    original_keyword: str
    acceptance_rate: float
    triggered: bool  # False if rate was acceptable
    alternative_keywords: list[str] = field(
        default_factory=list
    )  # keywords proposed by LLM
    new_rows_scraped: int = 0  # from the corrective scrape
    new_rows_accepted: int = 0  # accepted after enrichment


# --------------------------------------------------------------------------- #
# Acceptance rate computation
# --------------------------------------------------------------------------- #
def compute_acceptance_rates(
    run_id: str,
    profile_id: str | None,
    con: duckdb.DuckDBPyConnection,
) -> dict[str, float]:
    """Return {keyword: acceptance_rate} for all keywords in the given run.

    acceptance_rate = count(ensemble_verdict='accept') / count(all verdicts)
    for rows with the given run_id and profile_id.
    Returns {} if no verdicts exist yet.
    """
    params: list[object] = [run_id]
    profile_filter = ""
    if profile_id is not None:
        profile_filter = "AND j.profile_id = ?"
        params.append(profile_id)

    sql = f"""
        SELECT
            j.search_keyword,
            COUNT(*) AS total,
            SUM(CASE WHEN j.ensemble_verdict = 'accept' THEN 1 ELSE 0 END) AS accepted
        FROM judged_job_offers j
        JOIN staging_job_offers s ON s.id = j.job_id
        WHERE s.run_id = ?
          {profile_filter}
          AND j.search_keyword IS NOT NULL
        GROUP BY j.search_keyword
    """  # noqa: S608

    try:
        rows = con.execute(sql, params).fetchall()
    except duckdb.Error as exc:
        log.warning("corrective.compute_rates_failed", error=str(exc))
        return {}

    if not rows:
        return {}

    rates: dict[str, float] = {}
    for keyword, total, accepted in rows:
        if total and total > 0:
            rates[keyword] = accepted / total
        else:
            rates[keyword] = 0.0
    return rates


# --------------------------------------------------------------------------- #
# Sample titles for prompt construction
# --------------------------------------------------------------------------- #
def _sample_titles(
    run_id: str,
    profile_id: str | None,
    keyword: str,
    verdict: str,
    con: duckdb.DuckDBPyConnection,
    limit: int = 5,
) -> list[str]:
    """Return a sample of job titles for a given verdict and keyword."""
    if profile_id is not None:
        params: list[object] = [run_id, profile_id, keyword, verdict]
    else:
        params = [run_id, keyword, verdict]

    sql = f"""
        SELECT j.job_title
        FROM judged_job_offers j
        JOIN staging_job_offers s ON s.id = j.job_id
        WHERE s.run_id = ?
          {"AND j.profile_id = ?" if profile_id is not None else ""}
          AND j.search_keyword = ?
          AND j.ensemble_verdict = ?
          AND j.job_title IS NOT NULL
        ORDER BY j.judged_at DESC
        LIMIT {int(limit)}
    """  # noqa: S608

    try:
        rows = con.execute(sql, params).fetchall()
        return [r[0] for r in rows if r[0]]
    except duckdb.Error as exc:
        log.warning("corrective.sample_titles_failed", error=str(exc))
        return []


# --------------------------------------------------------------------------- #
# LLM alternative keyword proposal
# --------------------------------------------------------------------------- #
def propose_alternatives(
    keyword: str,
    accepted_titles: list[str],
    rejected_titles: list[str],
    *,
    settings: Settings,
    max_alternatives: int = C.CORRECTIVE_MAX_ALTERNATIVES,
) -> list[str]:
    """Ask the LLM to propose alternative search keywords.

    Returns a list of alternative keyword strings (may be empty on failure).
    """
    prompt = (
        f"I searched for job postings with keyword: '{keyword}'.\n"
        f"These titles were relevant: {accepted_titles[:5]}\n"
        f"These titles were NOT relevant: {rejected_titles[:5]}\n"
        f"Suggest {max_alternatives} alternative job search keywords "
        f"that would find more results like the relevant ones and fewer "
        f"like the irrelevant ones. "
        f'Return ONLY a JSON array of strings, e.g. ["phrase 1", "phrase 2"]. '
        f"No explanation, no markdown, just the JSON array."
    )

    try:
        if settings.llm.provider == C.PROVIDER_ANTHROPIC and settings.anthropic_api_key:
            import anthropic

            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
            response = client.messages.create(
                model=settings.llm.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
        else:
            import openai

            key = (
                settings.openai_api_key.get_secret_value()
                if settings.openai_api_key
                else "sk-not-needed"
            )
            client_kwargs: dict[str, object] = {"api_key": key}
            if settings.llm.base_url:
                client_kwargs["base_url"] = settings.llm.base_url
            client = openai.OpenAI(**client_kwargs)  # type: ignore[arg-type]
            response = client.chat.completions.create(
                model=settings.llm.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()

        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        log.warning("corrective.propose_not_list", keyword=keyword, raw=raw[:100])
        return []
    except Exception as exc:
        log.warning("corrective.propose_failed", keyword=keyword, error=str(exc))
        return []


# --------------------------------------------------------------------------- #
# Core corrective pass
# --------------------------------------------------------------------------- #
def run_corrective_pass(
    run_id: str,
    profile_id: str | None,
    sites: tuple[str, ...],
    settings: Settings,
    con: duckdb.DuckDBPyConnection,
) -> list[CorrectiveResult]:
    """For each keyword with acceptance rate below threshold, run one corrective pass.

    Steps per low-acceptance keyword:
    1. Get acceptance rate.
    2. Sample accepted/rejected titles for that keyword.
    3. Ask LLM for alternative keywords.
    4. Re-scrape with alternative keywords (same sites, same locations as original).
    5. Enrich the new staging rows.
    6. Return CorrectiveResult per keyword.

    Does NOT update the profile's saved keyword list — alternatives are
    ephemeral for this run. Returns [] if no keywords needed correction.
    """
    rates = compute_acceptance_rates(run_id, profile_id, con)
    if not rates:
        log.info("corrective.no_verdicts", run_id=run_id)
        return []

    # Find the original locations from this run so re-scrapes are consistent.
    try:
        loc_rows = con.execute(
            "SELECT DISTINCT search_location FROM staging_job_offers WHERE run_id = ?",
            [run_id],
        ).fetchall()
        locations: list[str | None] = [r[0] for r in loc_rows]
        if not locations:
            locations = [None]
    except duckdb.Error:
        locations = [None]

    results: list[CorrectiveResult] = []

    for keyword, rate in rates.items():
        if rate >= C.CORRECTIVE_ACCEPTANCE_THRESHOLD:
            results.append(
                CorrectiveResult(
                    original_keyword=keyword,
                    acceptance_rate=rate,
                    triggered=False,
                )
            )
            log.info(
                "corrective.skip",
                keyword=keyword,
                acceptance_rate=rate,
                threshold=C.CORRECTIVE_ACCEPTANCE_THRESHOLD,
            )
            continue

        log.info(
            "corrective.triggered",
            keyword=keyword,
            acceptance_rate=rate,
            threshold=C.CORRECTIVE_ACCEPTANCE_THRESHOLD,
        )

        # Sample titles to build a useful prompt.
        accepted_titles = _sample_titles(run_id, profile_id, keyword, "accept", con)
        rejected_titles = _sample_titles(run_id, profile_id, keyword, "reject", con)

        alternatives = propose_alternatives(
            keyword,
            accepted_titles,
            rejected_titles,
            settings=settings,
        )

        if not alternatives:
            log.info("corrective.no_alternatives", keyword=keyword)
            results.append(
                CorrectiveResult(
                    original_keyword=keyword,
                    acceptance_rate=rate,
                    triggered=True,
                    alternative_keywords=[],
                )
            )
            continue

        # Re-scrape with alternative keywords.
        alt_requests = [
            ScrapeRequest(keyword=alt_kw, location=loc, sites=sites)
            for alt_kw in alternatives
            for loc in locations
        ]

        new_rows_scraped = 0
        try:
            with connect() as scrape_con:
                scrape_results = scrape_to_staging(
                    run_id=run_id,
                    requests=alt_requests,
                    profile_id=profile_id,
                    con=scrape_con,
                )
            new_rows_scraped = sum(r.rows for r in scrape_results)
            log.info(
                "corrective.scraped",
                keyword=keyword,
                alternatives=alternatives,
                new_rows=new_rows_scraped,
            )
        except Exception as exc:
            log.warning("corrective.scrape_failed", keyword=keyword, error=str(exc))

        # Enrich new rows.
        new_rows_accepted = 0
        if new_rows_scraped > 0:
            try:
                provider = build_provider(settings.llm, settings)
                with connect() as enrich_con:
                    enrich_summary = enrich_staging(
                        run_id=run_id,
                        provider=provider,
                        con=enrich_con,
                    )
                # Count newly accepted rows for the alternative keywords.
                alt_keyword_set = set(alternatives)
                try:
                    acc_rows = con.execute(
                        """
                        SELECT COUNT(*)
                        FROM judged_job_offers j
                        JOIN staging_job_offers s ON s.id = j.job_id
                        WHERE s.run_id = ?
                          AND j.search_keyword = ANY(?)
                          AND j.ensemble_verdict = 'accept'
                        """,
                        [run_id, list(alt_keyword_set)],
                    ).fetchone()
                    new_rows_accepted = int(acc_rows[0]) if acc_rows else 0
                except duckdb.Error:
                    new_rows_accepted = enrich_summary.succeeded
            except Exception as exc:
                log.warning("corrective.enrich_failed", keyword=keyword, error=str(exc))

        results.append(
            CorrectiveResult(
                original_keyword=keyword,
                acceptance_rate=rate,
                triggered=True,
                alternative_keywords=alternatives,
                new_rows_scraped=new_rows_scraped,
                new_rows_accepted=new_rows_accepted,
            )
        )

    return results


__all__ = [
    "CorrectiveResult",
    "compute_acceptance_rates",
    "propose_alternatives",
    "run_corrective_pass",
]
