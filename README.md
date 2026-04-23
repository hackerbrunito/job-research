# job-research

Scrape, enrich, and analyze job listings across LinkedIn, Indeed, Glassdoor, and Google Jobs.

## Stack

Python 3.13 · `uv` · python-jobspy · Anthropic / OpenAI / Ollama · DuckDB · Prefect 3 · Streamlit · pydantic-settings · httpx · structlog.

## Quick start

```bash
uv sync
cp .env.example .env        # add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
uv run python -m job_research.pipeline
uv run streamlit run src/job_research/app/main.py
```

## Project layout

```
src/job_research/
├── config.py          # pydantic-settings (nested groups, SecretStr)
├── constants.py       # named constants — no magic values
├── database.py        # DuckDB connection, schema loader, key helpers
├── logging_setup.py   # structlog configuration
├── scraper.py         # JobSpy integration (coming)
├── enricher.py        # LLM enrichment orchestrator (coming)
├── llm_providers.py   # Anthropic / OpenAI / OpenAI-compatible (coming)
├── transform.py       # SQL transformations → marts (coming)
├── pipeline.py        # Prefect flow (coming)
└── app/               # Streamlit multi-page dashboard (coming)
sql/
├── ddl/               # CREATE TABLE statements (star schema)
└── dml/               # Upsert / refresh SQL (coming)
```

## Star schema

`staging_job_offers → int_enriched_job_info → {dim_location, dim_salary, dim_skill} → fact_job_offers + job_skill_bridge → mart_*`

