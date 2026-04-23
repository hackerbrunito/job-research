-- =========================================================================
-- Marts — denormalized aggregates for the dashboard. Refreshed by transform.
-- Implemented as tables (not views) so the dashboard loads instantly.
-- =========================================================================

CREATE TABLE IF NOT EXISTS mart_jobs_by_country (
    country_code   VARCHAR,
    country        VARCHAR,
    search_keyword VARCHAR,
    job_count      BIGINT NOT NULL,
    refreshed_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (country_code, search_keyword)
);

CREATE TABLE IF NOT EXISTS mart_skills_by_keyword (
    search_keyword VARCHAR,
    skill_name     VARCHAR,
    skill_type     VARCHAR,
    demand_count   BIGINT NOT NULL,
    refreshed_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (search_keyword, skill_type, skill_name)
);

CREATE TABLE IF NOT EXISTS mart_salary_by_keyword (
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
    PRIMARY KEY (search_keyword, currency, period)
);

CREATE TABLE IF NOT EXISTS mart_work_mode_distribution (
    search_keyword VARCHAR,
    work_mode      VARCHAR,
    job_count      BIGINT NOT NULL,
    refreshed_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (search_keyword, work_mode)
);
