-- =========================================================================
-- Fact — one row per enriched job offer. FKs to dimensions. Nulls allowed
-- where the enricher could not extract a value.
-- =========================================================================
CREATE TABLE IF NOT EXISTS fact_job_offers (
    job_id         VARCHAR PRIMARY KEY REFERENCES staging_job_offers(id),
    run_id         VARCHAR NOT NULL,
    profile_id     VARCHAR,                         -- tag: which saved search this came from
    scraped_at     TIMESTAMP NOT NULL,
    enriched_at    TIMESTAMP NOT NULL,
    site           VARCHAR NOT NULL,
    search_keyword VARCHAR NOT NULL,
    company        VARCHAR,
    title          VARCHAR,
    job_url        VARCHAR NOT NULL,
    date_posted    DATE,
    work_mode      VARCHAR,
    location_key   VARCHAR REFERENCES dim_location(location_key),
    salary_key     VARCHAR REFERENCES dim_salary(salary_key)
);

CREATE INDEX IF NOT EXISTS idx_fact_run_id      ON fact_job_offers(run_id);
CREATE INDEX IF NOT EXISTS idx_fact_site        ON fact_job_offers(site);
CREATE INDEX IF NOT EXISTS idx_fact_keyword     ON fact_job_offers(search_keyword);
CREATE INDEX IF NOT EXISTS idx_fact_date_posted ON fact_job_offers(date_posted);

-- Idempotent migration.
ALTER TABLE fact_job_offers ADD COLUMN IF NOT EXISTS profile_id VARCHAR;
CREATE INDEX IF NOT EXISTS idx_fact_profile ON fact_job_offers(profile_id);

-- =========================================================================
-- Bridge — many-to-many between job offers and skills.
-- =========================================================================
CREATE TABLE IF NOT EXISTS job_skill_bridge (
    job_id    VARCHAR NOT NULL REFERENCES fact_job_offers(job_id),
    skill_key VARCHAR NOT NULL REFERENCES dim_skill(skill_key),
    PRIMARY KEY (job_id, skill_key)
);

CREATE INDEX IF NOT EXISTS idx_bridge_skill ON job_skill_bridge(skill_key);
