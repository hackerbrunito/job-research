"""JobSpy-backed scraper that writes rows to staging_job_offers.

One request = (keyword, location, sites). Each site is scraped in an isolated
call so a 403/429 on one board cannot take down the others. Transient errors
are retried with exponential backoff via tenacity.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import duckdb
import pandas as pd
from jobspy import scrape_jobs
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from job_research.config import ScrapingConfig, get_settings
from job_research.constants import DEFAULT_SITES
from job_research.database import connect, insert_dataframe, job_id
from job_research.logging_setup import get_logger

log = get_logger(__name__)

STAGING_TABLE = "staging_job_offers"

# Columns we persist in staging, in the order expected by `raw_payload` serialization.
_STAGING_COLUMNS: tuple[str, ...] = (
    "id",
    "scraped_at",
    "run_id",
    "site",
    "search_keyword",
    "search_location",
    "job_url",
    "job_url_direct",
    "title",
    "company",
    "location_raw",
    "date_posted",
    "job_type",
    "salary_raw",
    "min_amount",
    "max_amount",
    "currency",
    "interval",
    "is_remote",
    "description",
    "company_url",
    "company_industry",
    "raw_payload",
)

# Transient errors worth retrying. We intentionally catch broad network/timeout
# errors because jobspy wraps many underlying libraries and does not expose a
# stable exception hierarchy.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
)


@dataclass(frozen=True)
class ScrapeRequest:
    keyword: str
    location: str | None = None
    sites: tuple[str, ...] = DEFAULT_SITES


@dataclass
class ScrapeResult:
    request: ScrapeRequest
    rows: int = 0
    per_site_counts: dict[str, int] = field(default_factory=dict)
    per_site_errors: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# JobSpy wrapper with retries
# --------------------------------------------------------------------------- #
def _scrape_one_site(
    *,
    site: str,
    keyword: str,
    location: str | None,
    cfg: ScrapingConfig,
) -> pd.DataFrame:
    """Call jobspy for a single site with tenacity-driven retries."""
    proxies = cfg.proxy_list() or None

    @retry(
        stop=stop_after_attempt(max(1, cfg.max_retries)),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        reraise=True,
    )
    def _call() -> pd.DataFrame:
        df = scrape_jobs(
            site_name=[site],
            search_term=keyword,
            location=location,
            results_wanted=cfg.max_results_per_site,
            hours_old=cfg.hours_old,
            linkedin_fetch_description=cfg.linkedin_fetch_description,
            description_format="markdown",
            proxies=proxies,
            verbose=1,
        )
        return df if df is not None else pd.DataFrame()

    return _call()


# --------------------------------------------------------------------------- #
# DataFrame → staging rows
# --------------------------------------------------------------------------- #
def _coerce_optional(row: pd.Series, key: str) -> Any:
    """Pull a column value from a row, returning None for NaN/missing."""
    if key not in row:
        return None
    val = row[key]
    if val is None:
        return None
    # pandas NaN / NaT detection without importing numpy
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _build_raw_payload(row: pd.Series) -> str:
    """JSON-serialize the full source row (description excluded to save space)."""
    payload: dict[str, Any] = {}
    for key in row.index:
        if key == "description":
            continue
        v = _coerce_optional(row, str(key))
        if v is None:
            payload[str(key)] = None
        elif isinstance(v, (str, int, float, bool)):
            payload[str(key)] = v
        else:
            payload[str(key)] = str(v)
    return json.dumps(payload, ensure_ascii=False, default=str)


def _build_staging_dataframe(
    df: pd.DataFrame,
    *,
    run_id: str,
    request: ScrapeRequest,
    scraped_at: datetime,
) -> pd.DataFrame:
    """Map a JobSpy DataFrame to the staging table schema."""
    if df.empty:
        return pd.DataFrame(columns=list(_STAGING_COLUMNS))

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        job_url = _coerce_optional(row, "job_url")
        if not job_url or not str(job_url).strip():
            continue
        site = _coerce_optional(row, "site")
        if not site:
            # Fall back to the first site in the request if jobspy didn't set one.
            site = request.sites[0] if request.sites else ""
        site = str(site)

        records.append(
            {
                "id": job_id(site, str(job_url)),
                "scraped_at": scraped_at,
                "run_id": run_id,
                "site": site,
                "search_keyword": request.keyword,
                "search_location": request.location,
                "job_url": str(job_url),
                "job_url_direct": _coerce_optional(row, "job_url_direct"),
                "title": _coerce_optional(row, "title"),
                "company": _coerce_optional(row, "company"),
                "location_raw": _coerce_optional(row, "location"),
                "date_posted": _coerce_optional(row, "date_posted"),
                "job_type": _coerce_optional(row, "job_type"),
                "salary_raw": _coerce_optional(row, "salary_source"),
                "min_amount": _coerce_optional(row, "min_amount"),
                "max_amount": _coerce_optional(row, "max_amount"),
                "currency": _coerce_optional(row, "currency"),
                "interval": _coerce_optional(row, "interval"),
                "is_remote": _coerce_optional(row, "is_remote"),
                "description": _coerce_optional(row, "description"),
                "company_url": _coerce_optional(row, "company_url"),
                "company_industry": _coerce_optional(row, "company_industry"),
                "raw_payload": _build_raw_payload(row),
            }
        )

    staging = pd.DataFrame.from_records(records, columns=list(_STAGING_COLUMNS))
    if not staging.empty:
        staging = staging.drop_duplicates(subset=["id"], keep="first")
    return staging


def _existing_ids(con: duckdb.DuckDBPyConnection, ids: Iterable[str]) -> set[str]:
    """Return ids already present in staging so we don't violate the PK."""
    id_list = [i for i in ids if i]
    if not id_list:
        return set()
    df = pd.DataFrame({"id": id_list})
    con.register("_ids_df", df)
    try:
        rows = con.execute(
            f"SELECT id FROM {STAGING_TABLE} WHERE id IN (SELECT id FROM _ids_df)"  # noqa: S608
        ).fetchall()
    finally:
        con.unregister("_ids_df")
    return {r[0] for r in rows}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def scrape_to_staging(
    run_id: str,
    requests: list[ScrapeRequest],
    *,
    con: duckdb.DuckDBPyConnection | None = None,
) -> list[ScrapeResult]:
    """Scrape each request, write rows to staging_job_offers, return results.

    Each (request, site) pair is a separate jobspy call so a failure on one
    site does not abort the others. A per-request result records row counts
    and any site-level errors.
    """
    if con is None:
        with connect() as owned_con:
            return _run(run_id, requests, owned_con)
    return _run(run_id, requests, con)


def _run(
    run_id: str,
    requests: list[ScrapeRequest],
    con: duckdb.DuckDBPyConnection,
) -> list[ScrapeResult]:
    cfg = get_settings().scraping
    results: list[ScrapeResult] = []

    for req in requests:
        log.info(
            "scrape.request.start",
            run_id=run_id,
            keyword=req.keyword,
            location=req.location,
            sites=list(req.sites),
        )
        result = ScrapeResult(request=req)

        for site in req.sites:
            try:
                df = _scrape_one_site(
                    site=site,
                    keyword=req.keyword,
                    location=req.location,
                    cfg=cfg,
                )
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                log.error(
                    "scrape.site.error",
                    run_id=run_id,
                    site=site,
                    keyword=req.keyword,
                    error=err,
                )
                result.per_site_errors[site] = err
                result.per_site_counts[site] = 0
                _polite_sleep(cfg.request_delay_seconds)
                continue

            staging_df = _build_staging_dataframe(
                df,
                run_id=run_id,
                request=req,
                scraped_at=datetime.now(UTC).replace(tzinfo=None),
            )

            # Filter rows whose ids already exist so PK conflicts don't abort insert.
            if not staging_df.empty:
                already = _existing_ids(con, staging_df["id"].tolist())
                if already:
                    staging_df = staging_df[~staging_df["id"].isin(already)]

            inserted = insert_dataframe(con, staging_df, STAGING_TABLE)
            result.per_site_counts[site] = inserted
            result.per_site_errors.setdefault(site, "")
            result.rows += inserted

            log.info(
                "scrape.site.done",
                run_id=run_id,
                site=site,
                keyword=req.keyword,
                rows_inserted=inserted,
            )
            _polite_sleep(cfg.request_delay_seconds)

        log.info(
            "scrape.request.done",
            run_id=run_id,
            keyword=req.keyword,
            rows=result.rows,
            per_site_counts=result.per_site_counts,
            errors={k: v for k, v in result.per_site_errors.items() if v},
        )
        results.append(result)

    return results


def _polite_sleep(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(seconds)
