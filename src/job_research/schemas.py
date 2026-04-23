"""Pydantic v2 schemas for LLM structured output.

These models are passed to provider SDKs as JSON schema (Anthropic tool-use
`input_schema`, OpenAI `response_format=<Model>`). The same models validate
the raw LLM response before it flows into `int_enriched_job_info`.

Only values validated here should make it into the database.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from job_research import constants as C


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
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("country_code must be a string or null")
        s = v.strip().upper()
        if not s:
            return None
        if len(s) != 2 or not s.isalpha():
            raise ValueError(
                f"country_code must be ISO 3166-1 alpha-2 (2 letters), got {v!r}"
            )
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
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("currency must be a string or null")
        s = v.strip().upper()
        if not s:
            return None
        if len(s) != 3 or not s.isalpha():
            raise ValueError(f"currency must be a 3-letter uppercase code, got {v!r}")
        return s

    @field_validator("period", mode="before")
    @classmethod
    def _normalise_period(cls, v: object) -> object:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("period must be a string or null")
        s = v.strip().lower()
        if not s:
            return None
        allowed = {
            C.SALARY_PERIOD_YEARLY,
            C.SALARY_PERIOD_MONTHLY,
            C.SALARY_PERIOD_HOURLY,
        }
        if s not in allowed:
            raise ValueError(f"period must be one of {sorted(allowed)}, got {v!r}")
        return s


# --------------------------------------------------------------------------- #
# Top-level enrichment
# --------------------------------------------------------------------------- #
class JobEnrichment(BaseModel):
    """Full structured enrichment for a single job posting."""

    tech_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Technical skills: tools, languages, frameworks, platforms. "
            "Lowercase, deduplicated."
        ),
    )
    soft_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Soft skills as lowercase adjectives (e.g. 'collaborative', "
            "'analytical'). Deduplicated."
        ),
    )
    location: LocationExtraction = Field(default_factory=LocationExtraction)
    work_mode: str | None = Field(
        default=None,
        description="One of: remote | hybrid | on-site. Null if unknown.",
    )
    salary: SalaryExtraction = Field(default_factory=SalaryExtraction)

    @field_validator("tech_skills", "soft_skills", mode="before")
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
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("work_mode must be a string or null")
        s = v.strip().lower()
        if not s:
            return None
        # Accept a couple of common variants but reject everything else.
        aliases = {
            "onsite": C.WORK_MODE_ONSITE,
            "on site": C.WORK_MODE_ONSITE,
            "on_site": C.WORK_MODE_ONSITE,
        }
        s = aliases.get(s, s)
        if s not in C.WORK_MODES:
            raise ValueError(
                f"work_mode must be one of {sorted(C.WORK_MODES)}, got {v!r}"
            )
        return s
