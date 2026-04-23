-- =========================================================================
-- User search configuration — single-row-per-key JSON values used by the
-- dashboard's Search config page. The Run page reads these to know what
-- to scrape.
--
-- Expected keys:
--   'keywords'  — JSON array of strings
--   'locations' — JSON array of strings
--   'sites'     — JSON array of strings (subset of constants.ALL_SITES)
-- =========================================================================
CREATE TABLE IF NOT EXISTS user_search_config (
    key        VARCHAR PRIMARY KEY,
    value      JSON NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
