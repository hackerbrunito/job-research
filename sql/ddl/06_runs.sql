-- =========================================================================
-- Pipeline runs — one row per Prefect flow execution. Used by the
-- dashboard's History page.
-- =========================================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        VARCHAR PRIMARY KEY,
    started_at    TIMESTAMP NOT NULL,
    finished_at   TIMESTAMP,
    status        VARCHAR NOT NULL,       -- running | success | failed
    keywords      JSON,                   -- array of strings
    locations     JSON,                   -- array of strings
    sites         JSON,                   -- array of strings
    scraped_count BIGINT DEFAULT 0,
    enriched_count BIGINT DEFAULT 0,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_started ON pipeline_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON pipeline_runs(status);
