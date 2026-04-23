-- =========================================================================
-- Marts — denormalized aggregates for the dashboard. Refreshed by transform.
-- Implemented as tables (not views) so the dashboard loads instantly.
-- =========================================================================

-- Marts include profile_id so the dashboard can aggregate per-profile.
-- profile_id is stored as empty-string-coalesced so it can be part of the
-- PK (DuckDB PKs reject NULL).
CREATE TABLE IF NOT EXISTS mart_jobs_by_country (
    profile_id     VARCHAR NOT NULL DEFAULT '',
    country_code   VARCHAR,
    country        VARCHAR,
    search_keyword VARCHAR,
    job_count      BIGINT NOT NULL,
    refreshed_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (profile_id, country_code, search_keyword)
);

CREATE TABLE IF NOT EXISTS mart_skills_by_keyword (
    profile_id     VARCHAR NOT NULL DEFAULT '',
    search_keyword VARCHAR,
    skill_name     VARCHAR,
    skill_type     VARCHAR,
    demand_count   BIGINT NOT NULL,
    refreshed_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (profile_id, search_keyword, skill_type, skill_name)
);

CREATE TABLE IF NOT EXISTS mart_salary_by_keyword (
    profile_id        VARCHAR NOT NULL DEFAULT '',
    search_keyword    VARCHAR,
    currency          VARCHAR,
    period            VARCHAR,
    p25_min           DOUBLE,
    p50_min           DOUBLE,
    p75_min           DOUBLE,
    p25_max           DOUBLE,
    p50_max           DOUBLE,
    p75_max           DOUBLE,
    observation_count BIGINT NOT NULL,
    refreshed_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (profile_id, search_keyword, currency, period)
);

CREATE TABLE IF NOT EXISTS mart_work_mode_distribution (
    profile_id     VARCHAR NOT NULL DEFAULT '',
    search_keyword VARCHAR,
    work_mode      VARCHAR,
    job_count      BIGINT NOT NULL,
    refreshed_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (profile_id, search_keyword, work_mode)
);

-- Idempotent migration for existing mart tables (pre-profile_id).
-- DuckDB rejects PK changes via ALTER, so we just add the column; the
-- PK still reflects the original shape until the table is dropped and
-- rebuilt. Marts are full-refreshed by transform.run_transform() so the
-- easiest upgrade path is `rm data/jobs.duckdb` — documented in README.
ALTER TABLE mart_jobs_by_country        ADD COLUMN IF NOT EXISTS profile_id VARCHAR;
ALTER TABLE mart_skills_by_keyword      ADD COLUMN IF NOT EXISTS profile_id VARCHAR;
ALTER TABLE mart_salary_by_keyword      ADD COLUMN IF NOT EXISTS profile_id VARCHAR;
ALTER TABLE mart_work_mode_distribution ADD COLUMN IF NOT EXISTS profile_id VARCHAR;
