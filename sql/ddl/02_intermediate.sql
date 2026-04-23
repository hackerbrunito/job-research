-- =========================================================================
-- Intermediate — LLM-enriched structured data. Joined back to staging by id.
-- One row per staging row that was successfully enriched.
-- =========================================================================
CREATE TABLE IF NOT EXISTS int_enriched_job_info (
    job_id          VARCHAR PRIMARY KEY REFERENCES staging_job_offers(id),
    enriched_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    llm_provider    VARCHAR NOT NULL,   -- anthropic | openai | openai-compatible
    llm_model       VARCHAR NOT NULL,
    tech_skills     JSON NOT NULL,      -- array of lowercase strings
    soft_skills     JSON NOT NULL,      -- array of lowercase adjective-form strings
    city            VARCHAR,
    country         VARCHAR,
    country_code    VARCHAR,            -- ISO 3166-1 alpha-2
    work_mode       VARCHAR,            -- remote | hybrid | on-site
    salary_min      DOUBLE,
    salary_max      DOUBLE,
    salary_currency VARCHAR,            -- ISO 4217 or 3-letter code
    salary_period   VARCHAR,            -- yearly | monthly | hourly
    raw_response    JSON                -- full LLM output for forensics
);

CREATE INDEX IF NOT EXISTS idx_int_enriched_country ON int_enriched_job_info(country_code);
CREATE INDEX IF NOT EXISTS idx_int_enriched_mode    ON int_enriched_job_info(work_mode);
