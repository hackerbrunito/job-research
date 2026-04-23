-- =========================================================================
-- User search profiles — one row per named search the user has saved.
-- Multiple profiles support searching for distinct personas (e.g. one
-- profile for data-engineering roles, another for retail roles).
--
-- Legacy single-row table `user_search_config` is preserved so older
-- databases survive; the app reads/writes only `user_search_profiles`.
-- =========================================================================

-- Legacy table (kept for backwards-compat on old DBs).
CREATE TABLE IF NOT EXISTS user_search_config (
    key        VARCHAR PRIMARY KEY,
    value      JSON NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- New profiles table.
CREATE TABLE IF NOT EXISTS user_search_profiles (
    profile_id   VARCHAR PRIMARY KEY,       -- slug form of name
    name         VARCHAR NOT NULL,          -- display name
    description  VARCHAR,                   -- optional free text
    keywords     JSON NOT NULL,             -- array of strings
    locations    JSON NOT NULL,             -- array of strings
    sites        JSON NOT NULL,             -- array of strings (subset of ALL_SITES)
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_profiles_name ON user_search_profiles(name);
