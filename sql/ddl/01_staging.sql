-- =========================================================================
-- Staging — raw scraped job offers, one row per JobSpy result.
-- Unenriched; source for the enrichment LLM.
-- =========================================================================
CREATE TABLE IF NOT EXISTS staging_job_offers (
    id               VARCHAR PRIMARY KEY,           -- deterministic hash(site, job_url)
    scraped_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    run_id           VARCHAR NOT NULL,              -- pipeline run that scraped this row
    site             VARCHAR NOT NULL,              -- linkedin | indeed | glassdoor | google | zip_recruiter
    search_keyword   VARCHAR NOT NULL,              -- search term that produced this result
    search_location  VARCHAR,                       -- location filter used at search time
    job_url          VARCHAR NOT NULL,
    job_url_direct   VARCHAR,
    title            VARCHAR,
    company          VARCHAR,
    location_raw     VARCHAR,                       -- free-text location from source
    date_posted      DATE,
    job_type         VARCHAR,                       -- fulltime | parttime | contract | internship
    salary_raw       VARCHAR,                       -- free-text salary from source
    min_amount       DOUBLE,                        -- from jobspy if available (pre-enrichment)
    max_amount       DOUBLE,
    currency         VARCHAR,
    interval         VARCHAR,                       -- yearly | monthly | hourly
    is_remote        BOOLEAN,                       -- from jobspy if available (pre-enrichment)
    description      TEXT,
    company_url      VARCHAR,
    company_industry VARCHAR,
    raw_payload      JSON                           -- full source record for forensic replay
);

CREATE INDEX IF NOT EXISTS idx_staging_run_id   ON staging_job_offers(run_id);
CREATE INDEX IF NOT EXISTS idx_staging_site     ON staging_job_offers(site);
CREATE INDEX IF NOT EXISTS idx_staging_scraped  ON staging_job_offers(scraped_at);
