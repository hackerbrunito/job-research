"""Pydantic v2 schemas for LLM structured output.

These models are passed to provider SDKs as JSON schema (Anthropic tool-use
`input_schema`, OpenAI `response_format=<Model>`). The same models validate
the raw LLM response before it flows into `int_enriched_job_info`.

Only values validated here should make it into the database.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from job_research import constants as C

# Values the LLM sometimes returns for "no data" that should be treated as
# an empty object, not a hard validation failure. Anthropic has been
# observed returning the literal string "null" for sub-objects.
_EMPTY_SENTINELS: frozenset[str] = frozenset({"", "null", "none", "unknown", "n/a"})


def _is_empty_sentinel(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in _EMPTY_SENTINELS:
        return True
    return isinstance(v, dict) and not v


# --------------------------------------------------------------------------- #
# Sub-models
# --------------------------------------------------------------------------- #
class LocationExtraction(BaseModel):
    """City / country extracted from the job posting."""

    city: str | None = None
    country: str | None = None
    country_code: str | None = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code, uppercase (e.g. US, GB, ES).",
    )

    @field_validator("city", "country", mode="before")
    @classmethod
    def _strip_or_none(cls, v: object) -> object:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s or None
        return v

    @field_validator("country_code", mode="before")
    @classmethod
    def _normalise_country_code(cls, v: object) -> object:
        """Coerce to ISO 3166-1 alpha-2 or None. Malformed values drop to None
        rather than failing the whole enrichment."""
        if v is None or not isinstance(v, str):
            return None
        s = v.strip().upper()
        if not s or len(s) != 2 or not s.isalpha():
            return None
        return s


class SalaryExtraction(BaseModel):
    """Salary band extracted from the job posting."""

    min_amount: float | None = None
    max_amount: float | None = None
    currency: str | None = Field(
        default=None,
        description="3-letter uppercase currency code (ISO 4217) e.g. USD, EUR, GBP.",
    )
    period: str | None = Field(
        default=None,
        description="One of: yearly | monthly | hourly.",
    )

    @field_validator("currency", mode="before")
    @classmethod
    def _normalise_currency(cls, v: object) -> object:
        """Coerce to 3-letter ISO 4217 or None. Malformed drops to None."""
        if v is None or not isinstance(v, str):
            return None
        s = v.strip().upper()
        if not s or len(s) != 3 or not s.isalpha():
            return None
        return s

    @field_validator("period", mode="before")
    @classmethod
    def _normalise_period(cls, v: object) -> object:
        """Unknown periods (e.g. 'daily', 'weekly') coerce to None rather than
        failing the whole enrichment. We'd rather keep the min/max/currency
        signal and drop just the period than lose the row."""
        if v is None:
            return None
        if not isinstance(v, str):
            return None
        s = v.strip().lower()
        if not s:
            return None
        allowed = {
            C.SALARY_PERIOD_YEARLY,
            C.SALARY_PERIOD_MONTHLY,
            C.SALARY_PERIOD_HOURLY,
        }
        return s if s in allowed else None


# --------------------------------------------------------------------------- #
# Top-level enrichment
# --------------------------------------------------------------------------- #
class JobEnrichment(BaseModel):
    """Full structured enrichment for a single job posting."""

    tech_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Technology skills ONLY: concrete tools, programming languages, "
            "frameworks, libraries, platforms, cloud services, databases, "
            "protocols. Examples: python, spark, aws, kubernetes, postgres, "
            "react. Lowercase, deduplicated. Do NOT put methodologies, "
            "soft skills, or domain practices here."
        ),
    )
    soft_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Interpersonal / character traits as lowercase adjectives. "
            "Examples: collaborative, analytical, proactive, resilient, "
            "detail-oriented. Deduplicated. NOT job functions or tools."
        ),
    )
    domain_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Every other skill or qualification the posting explicitly "
            "lists that does NOT fit tech_skills (a technology/tool) or "
            "soft_skills (an interpersonal adjective). This includes "
            "domain expertise, business practices, methodologies, "
            "responsibilities, certifications, and industry-specific "
            "know-how. Examples: visual merchandising, sap retail, "
            "supplier negotiation, p&l ownership, agile delivery, "
            "stakeholder management, technical leadership, store layout "
            "design, budget management, threat modelling, penetration "
            "testing, mlops, project management, gdpr compliance. "
            "Lowercase, deduplicated, short phrases preferred."
        ),
    )
    location: LocationExtraction = Field(default_factory=LocationExtraction)
    work_mode: str | None = Field(
        default=None,
        description="One of: remote | hybrid | on-site. Null if unknown.",
    )
    salary: SalaryExtraction = Field(default_factory=SalaryExtraction)

    @field_validator("location", "salary", mode="before")
    @classmethod
    def _coerce_empty_sub_object(cls, v: object) -> object:
        """Accept the LLM's many ways of saying 'no data' for a sub-object.

        Anthropic has been observed emitting the literal string 'null' and
        OpenAI sometimes emits an empty dict. Treat both as the empty model.
        """
        if _is_empty_sentinel(v):
            return {}
        return v

    @field_validator("tech_skills", "soft_skills", "domain_skills", mode="before")
    @classmethod
    def _clean_skills(cls, v: object) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("skills must be a list of strings")
        seen: set[str] = set()
        out: list[str] = []
        for item in v:
            if item is None:
                continue
            if not isinstance(item, str):
                raise ValueError(f"skill entries must be strings, got {item!r}")
            s = item.strip().lower()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    @field_validator("work_mode", mode="before")
    @classmethod
    def _normalise_work_mode(cls, v: object) -> object:
        """Unknown / unparseable work_mode values coerce to None rather than
        fail the whole enrichment. Known aliases are canonicalized."""
        if v is None or not isinstance(v, str):
            return None
        s = v.strip().lower()
        if not s:
            return None
        aliases = {
            "onsite": C.WORK_MODE_ONSITE,
            "on site": C.WORK_MODE_ONSITE,
            "on_site": C.WORK_MODE_ONSITE,
            "in-office": C.WORK_MODE_ONSITE,
            "in office": C.WORK_MODE_ONSITE,
            "office": C.WORK_MODE_ONSITE,
        }
        s = aliases.get(s, s)
        return s if s in C.WORK_MODES else None
