"""Tests for enrichment schemas, provider factory, and `enrich_staging`."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import duckdb
import pytest
from pydantic import SecretStr, ValidationError

from job_research import constants as C
from job_research.config import LLMConfig, Settings
from job_research.enricher import EnrichmentSummary, enrich_staging
from job_research.llm_providers import build_provider
from job_research.schemas import (
    JobEnrichment,
    LocationExtraction,
    SalaryExtraction,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _FakeProvider:
    """In-memory provider for enricher tests."""

    provider_name = "fake"
    model_name = "fake-1"

    def __init__(self, responses: dict[str, JobEnrichment | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def enrich(self, *, title: str, description: str) -> JobEnrichment:
        self.calls.append((title, description))
        key = title.strip() or description[:20]
        value = self.responses[key]
        if isinstance(value, Exception):
            raise value
        return value


def _insert_staging(
    con: duckdb.DuckDBPyConnection,
    *,
    job_id: str,
    title: str,
    description: str,
    run_id: str = "run-1",
    site: str = C.SITE_LINKEDIN,
) -> None:
    con.execute(
        """
        INSERT INTO staging_job_offers
            (id, scraped_at, run_id, site, search_keyword, job_url,
             title, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            job_id,
            datetime.utcnow(),
            run_id,
            site,
            "python",
            f"https://example.test/{job_id}",
            title,
            description,
        ],
    )


def _make_enrichment(
    *,
    tech: list[str] | None = None,
    country_code: str | None = "US",
    work_mode: str | None = C.WORK_MODE_REMOTE,
) -> JobEnrichment:
    return JobEnrichment(
        tech_skills=tech or ["python"],
        soft_skills=["collaborative"],
        location=LocationExtraction(
            city="Remote", country="United States", country_code=country_code
        ),
        work_mode=work_mode,
        salary=SalaryExtraction(
            min_amount=100000.0,
            max_amount=140000.0,
            currency="USD",
            period=C.SALARY_PERIOD_YEARLY,
        ),
    )


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
class TestSchemaValidators:
    def test_tech_skills_lowercased_and_deduped(self) -> None:
        enrichment = JobEnrichment(
            tech_skills=["Python", "python", "SQL", " sql ", "Docker"],
            soft_skills=[],
        )
        assert enrichment.tech_skills == ["python", "sql", "docker"]

    def test_soft_skills_lowercased_and_deduped(self) -> None:
        enrichment = JobEnrichment(
            tech_skills=[],
            soft_skills=["Collaborative", "analytical", "analytical"],
        )
        assert enrichment.soft_skills == ["collaborative", "analytical"]

    def test_country_code_uppercased(self) -> None:
        loc = LocationExtraction(country_code="gb")
        assert loc.country_code == "GB"

    def test_country_code_rejects_non_alpha2(self) -> None:
        with pytest.raises(ValidationError):
            LocationExtraction(country_code="USA")

    def test_currency_uppercased(self) -> None:
        sal = SalaryExtraction(currency="eur")
        assert sal.currency == "EUR"

    def test_currency_rejects_invalid_length(self) -> None:
        with pytest.raises(ValidationError):
            SalaryExtraction(currency="DOLLARS")

    def test_period_rejects_invalid_value(self) -> None:
        with pytest.raises(ValidationError):
            SalaryExtraction(period="weekly")

    def test_work_mode_rejects_wfh(self) -> None:
        with pytest.raises(ValidationError):
            JobEnrichment(
                tech_skills=[],
                soft_skills=[],
                work_mode="WFH",
            )

    def test_work_mode_accepts_onsite_alias(self) -> None:
        enrichment = JobEnrichment(
            tech_skills=[],
            soft_skills=[],
            work_mode="onsite",
        )
        assert enrichment.work_mode == C.WORK_MODE_ONSITE

    def test_work_mode_none_is_ok(self) -> None:
        enrichment = JobEnrichment(
            tech_skills=[],
            soft_skills=[],
            work_mode=None,
        )
        assert enrichment.work_mode is None


# --------------------------------------------------------------------------- #
# enrich_staging
# --------------------------------------------------------------------------- #
class TestEnrichStaging:
    def test_happy_path_enriches_all_rows(
        self, tmp_duckdb: duckdb.DuckDBPyConnection
    ) -> None:
        _insert_staging(
            tmp_duckdb,
            job_id="j1",
            title="Python Dev",
            description="Build APIs",
        )
        _insert_staging(
            tmp_duckdb,
            job_id="j2",
            title="Data Engineer",
            description="Pipelines",
        )
        _insert_staging(
            tmp_duckdb,
            job_id="j3",
            title="ML Engineer",
            description="Models",
        )

        provider = _FakeProvider(
            {
                "Python Dev": _make_enrichment(tech=["python"]),
                "Data Engineer": _make_enrichment(tech=["sql", "airflow"]),
                "ML Engineer": _make_enrichment(tech=["pytorch"]),
            }
        )

        summary = enrich_staging(provider=provider, con=tmp_duckdb)

        assert summary == EnrichmentSummary(attempted=3, succeeded=3, failed=0)
        count = tmp_duckdb.execute(
            "SELECT COUNT(*) FROM int_enriched_job_info"
        ).fetchone()[0]
        assert count == 3

        # Confirm one row round-trips correctly.
        row = tmp_duckdb.execute(
            "SELECT job_id, llm_provider, llm_model, country_code, work_mode "
            "FROM int_enriched_job_info WHERE job_id = 'j1'"
        ).fetchone()
        assert row == ("j1", "fake", "fake-1", "US", C.WORK_MODE_REMOTE)

    def test_idempotent_skips_already_enriched(
        self, tmp_duckdb: duckdb.DuckDBPyConnection
    ) -> None:
        _insert_staging(tmp_duckdb, job_id="j1", title="A", description="desc-a")
        _insert_staging(tmp_duckdb, job_id="j2", title="B", description="desc-b")

        provider = _FakeProvider(
            {
                "A": _make_enrichment(),
                "B": _make_enrichment(),
            }
        )

        first = enrich_staging(provider=provider, con=tmp_duckdb)
        assert first.succeeded == 2

        # Second run: no new staging rows, so nothing to do.
        second = enrich_staging(provider=provider, con=tmp_duckdb)
        assert second == EnrichmentSummary(attempted=0, succeeded=0, failed=0)

        # Provider was NOT called again.
        assert len(provider.calls) == 2

        # Add a new row; only the new one should be enriched.
        _insert_staging(tmp_duckdb, job_id="j3", title="C", description="desc-c")
        provider.responses["C"] = _make_enrichment()
        third = enrich_staging(provider=provider, con=tmp_duckdb)
        assert third == EnrichmentSummary(attempted=1, succeeded=1, failed=0)

    def test_continues_after_provider_exception(
        self, tmp_duckdb: duckdb.DuckDBPyConnection
    ) -> None:
        _insert_staging(tmp_duckdb, job_id="j1", title="A", description="a")
        _insert_staging(tmp_duckdb, job_id="j2", title="B", description="b")
        _insert_staging(tmp_duckdb, job_id="j3", title="C", description="c")

        provider = _FakeProvider(
            {
                "A": _make_enrichment(),
                "B": RuntimeError("boom — simulated provider failure"),
                "C": _make_enrichment(),
            }
        )

        summary = enrich_staging(provider=provider, con=tmp_duckdb)
        assert summary == EnrichmentSummary(attempted=3, succeeded=2, failed=1)

        stored_ids = {
            r[0]
            for r in tmp_duckdb.execute(
                "SELECT job_id FROM int_enriched_job_info"
            ).fetchall()
        }
        assert stored_ids == {"j1", "j3"}

    def test_enricher_counts_validation_error_as_failed(
        self, tmp_duckdb: duckdb.DuckDBPyConnection
    ) -> None:
        """ValidationError from provider is caught and counted as failed."""
        _insert_staging(tmp_duckdb, job_id="j1", title="A", description="a")
        _insert_staging(tmp_duckdb, job_id="j2", title="B", description="b")
        _insert_staging(tmp_duckdb, job_id="j3", title="C", description="c")

        # Trigger a real ValidationError by attempting to validate a bad payload.
        try:
            JobEnrichment.model_validate({"tech_skills": "not-a-list"})
        except ValidationError as exc:
            validation_error = exc
        else:  # pragma: no cover — sanity
            raise AssertionError("expected ValidationError")

        provider = _FakeProvider(
            {
                "A": _make_enrichment(),
                "B": validation_error,
                "C": _make_enrichment(),
            }
        )

        summary = enrich_staging(provider=provider, con=tmp_duckdb)
        assert summary == EnrichmentSummary(attempted=3, succeeded=2, failed=1)

        stored_ids = {
            r[0]
            for r in tmp_duckdb.execute(
                "SELECT job_id FROM int_enriched_job_info"
            ).fetchall()
        }
        assert stored_ids == {"j1", "j3"}

    def test_limit_caps_attempted_rows(
        self, tmp_duckdb: duckdb.DuckDBPyConnection
    ) -> None:
        for i in range(5):
            _insert_staging(
                tmp_duckdb,
                job_id=f"j{i}",
                title=f"T{i}",
                description=f"d{i}",
            )

        provider = _FakeProvider({f"T{i}": _make_enrichment() for i in range(5)})

        summary = enrich_staging(provider=provider, con=tmp_duckdb, limit=2)
        assert summary.attempted == 2
        assert summary.succeeded == 2

    def test_run_id_filter(self, tmp_duckdb: duckdb.DuckDBPyConnection) -> None:
        _insert_staging(
            tmp_duckdb, job_id="j1", title="A", description="a", run_id="r1"
        )
        _insert_staging(
            tmp_duckdb, job_id="j2", title="B", description="b", run_id="r2"
        )

        provider = _FakeProvider({"A": _make_enrichment(), "B": _make_enrichment()})

        summary = enrich_staging(run_id="r1", provider=provider, con=tmp_duckdb)
        assert summary.attempted == 1
        assert summary.succeeded == 1
        # Only the r1 job landed.
        ids = {
            r[0]
            for r in tmp_duckdb.execute(
                "SELECT job_id FROM int_enriched_job_info"
            ).fetchall()
        }
        assert ids == {"j1"}


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #
class TestBuildProvider:
    def _settings(
        self,
        *,
        anthropic_key: str | None = None,
        openai_key: str | None = None,
    ) -> Settings:
        # Build a Settings instance without reading the real .env.
        with patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            s = Settings(
                anthropic_api_key=(SecretStr(anthropic_key) if anthropic_key else None),
                openai_api_key=(SecretStr(openai_key) if openai_key else None),
            )
        return s

    def test_anthropic_requires_api_key(self) -> None:
        cfg = LLMConfig(provider=C.PROVIDER_ANTHROPIC)
        settings = self._settings()  # no keys
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            build_provider(cfg, settings)

    def test_openai_requires_api_key(self) -> None:
        cfg = LLMConfig(provider=C.PROVIDER_OPENAI, model=C.DEFAULT_OPENAI_MODEL)
        settings = self._settings()
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            build_provider(cfg, settings)

    def test_openai_compatible_requires_base_url(self) -> None:
        cfg = LLMConfig(
            provider=C.PROVIDER_OPENAI_COMPATIBLE,
            model=C.DEFAULT_LOCAL_MODEL,
        )
        settings = self._settings()
        with pytest.raises(ValueError, match="base_url"):
            build_provider(cfg, settings)

    def test_anthropic_builds_with_key(self) -> None:
        cfg = LLMConfig(provider=C.PROVIDER_ANTHROPIC)
        settings = self._settings(anthropic_key="sk-ant-test")
        provider = build_provider(cfg, settings)
        assert provider.provider_name == C.PROVIDER_ANTHROPIC
        assert provider.model_name == C.DEFAULT_ANTHROPIC_MODEL

    def test_openai_compatible_builds_without_key(self) -> None:
        cfg = LLMConfig(
            provider=C.PROVIDER_OPENAI_COMPATIBLE,
            model=C.DEFAULT_LOCAL_MODEL,
            base_url="http://localhost:11434/v1",
        )
        settings = self._settings()  # no openai key
        provider = build_provider(cfg, settings)
        assert provider.provider_name == C.PROVIDER_OPENAI_COMPATIBLE
