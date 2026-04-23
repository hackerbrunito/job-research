"""Named constants for job-research.

All magic values go here — timeouts, limits, defaults, supported options.
Import from here; do not re-define literal values in other modules.
"""

from __future__ import annotations

from typing import Final

# ---- Application ---------------------------------------------------------
APP_NAME: Final[str] = "job-research"
APP_VERSION: Final[str] = "0.1.0"

# ---- Supported LLM providers --------------------------------------------
PROVIDER_ANTHROPIC: Final[str] = "anthropic"
PROVIDER_OPENAI: Final[str] = "openai"
PROVIDER_OPENAI_COMPATIBLE: Final[str] = "openai-compatible"
SUPPORTED_PROVIDERS: Final[frozenset[str]] = frozenset(
    {PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_OPENAI_COMPATIBLE}
)

# ---- Default model IDs (April 2026, verified) ---------------------------
DEFAULT_ANTHROPIC_MODEL: Final[str] = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_LOCAL_MODEL: Final[str] = "llama3:8b"

# ---- JobSpy sites --------------------------------------------------------
SITE_LINKEDIN: Final[str] = "linkedin"
SITE_INDEED: Final[str] = "indeed"
SITE_GLASSDOOR: Final[str] = "glassdoor"
SITE_GOOGLE: Final[str] = "google"
SITE_ZIPRECRUITER: Final[str] = "zip_recruiter"
# Glassdoor intentionally excluded from defaults: jobspy routinely fails to
# resolve non-US locations ("Glassdoor: location not parsed") and the API
# aggressively blocks from non-US IPs. Still available via opt-in on the
# Search page or ALL_SITES if the user wants to try it.
DEFAULT_SITES: Final[tuple[str, ...]] = (SITE_LINKEDIN, SITE_INDEED)
ALL_SITES: Final[tuple[str, ...]] = (
    SITE_LINKEDIN,
    SITE_INDEED,
    SITE_GLASSDOOR,
    SITE_GOOGLE,
    SITE_ZIPRECRUITER,
)

# ---- Work modes ----------------------------------------------------------
WORK_MODE_REMOTE: Final[str] = "remote"
WORK_MODE_HYBRID: Final[str] = "hybrid"
WORK_MODE_ONSITE: Final[str] = "on-site"
WORK_MODES: Final[frozenset[str]] = frozenset(
    {WORK_MODE_REMOTE, WORK_MODE_HYBRID, WORK_MODE_ONSITE}
)

# ---- Skill types ---------------------------------------------------------
SKILL_TYPE_TECH: Final[str] = "tech"
SKILL_TYPE_SOFT: Final[str] = "soft"
# `domain` covers everything a job posting lists under "skills" /
# "qualifications" that is neither a technology nor an interpersonal
# adjective. Examples: "visual merchandising", "SAP Retail", "supplier
# negotiation", "P&L ownership", "agile delivery", "technical leadership".
SKILL_TYPE_DOMAIN: Final[str] = "domain"
SKILL_TYPES: Final[frozenset[str]] = frozenset(
    {SKILL_TYPE_TECH, SKILL_TYPE_SOFT, SKILL_TYPE_DOMAIN}
)

# ---- Search profiles -----------------------------------------------------
DEFAULT_PROFILE_ID: Final[str] = "default"
DEFAULT_PROFILE_NAME: Final[str] = "Default"

# ---- Salary periods ------------------------------------------------------
SALARY_PERIOD_YEARLY: Final[str] = "yearly"
SALARY_PERIOD_MONTHLY: Final[str] = "monthly"
SALARY_PERIOD_HOURLY: Final[str] = "hourly"

# ---- Scraping defaults ---------------------------------------------------
DEFAULT_RESULTS_PER_SITE: Final[int] = 25
DEFAULT_HOURS_OLD: Final[int] = 72
DEFAULT_REQUEST_DELAY_SECONDS: Final[float] = 1.5
DEFAULT_SCRAPING_MAX_RETRIES: Final[int] = 3
DEFAULT_SCRAPING_TIMEOUT_SECONDS: Final[int] = 30

# ---- LLM defaults --------------------------------------------------------
DEFAULT_LLM_MAX_TOKENS: Final[int] = 2048
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.0
DEFAULT_LLM_TIMEOUT_SECONDS: Final[int] = 60
DEFAULT_LLM_MAX_RETRIES: Final[int] = 2
ENRICH_BATCH_SIZE: Final[int] = 10

# ---- Database ------------------------------------------------------------
DEFAULT_DB_PATH: Final[str] = "data/jobs.duckdb"

# ---- Prefect -------------------------------------------------------------
FLOW_RETRY_COUNT: Final[int] = 2
FLOW_RETRY_DELAY_SECONDS: Final[int] = 30

# ---- Dashboard -----------------------------------------------------------
DEFAULT_DASHBOARD_HOST: Final[str] = "localhost"
DEFAULT_DASHBOARD_PORT: Final[int] = 8501
