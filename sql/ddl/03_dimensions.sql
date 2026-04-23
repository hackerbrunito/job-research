-- =========================================================================
-- Dimensions — small lookup tables referenced by the fact table.
-- Deterministic surrogate keys so upserts are idempotent.
-- =========================================================================

-- Location: city + country_code is the natural key.
CREATE TABLE IF NOT EXISTS dim_location (
    location_key VARCHAR PRIMARY KEY,    -- hash(lower(city), country_code)
    city         VARCHAR,
    country      VARCHAR,
    country_code VARCHAR                 -- ISO 3166-1 alpha-2
);

-- Salary bucket: min + max + currency + period is the natural key.
CREATE TABLE IF NOT EXISTS dim_salary (
    salary_key  VARCHAR PRIMARY KEY,     -- hash(min, max, currency, period)
    min_amount  DOUBLE,
    max_amount  DOUBLE,
    currency    VARCHAR,
    period      VARCHAR                  -- yearly | monthly | hourly
);

-- Skill: lowercase name + type is the natural key.
CREATE TABLE IF NOT EXISTS dim_skill (
    skill_key  VARCHAR PRIMARY KEY,      -- hash(name, skill_type)
    name       VARCHAR NOT NULL,
    skill_type VARCHAR NOT NULL          -- tech | soft
);

CREATE INDEX IF NOT EXISTS idx_dim_skill_type ON dim_skill(skill_type);
