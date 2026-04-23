-- =========================================================================
-- Verdict layer — one row per staging row. Written in two passes:
--   Pass 1 (scraper / Agent 1): rule_verdict and initial ensemble_verdict.
--   Pass 2 (enricher / Agent 2): llm_* columns and recomputed ensemble_verdict.
-- =========================================================================
CREATE TABLE IF NOT EXISTS judged_job_offers (
    job_id              VARCHAR PRIMARY KEY REFERENCES staging_job_offers(id),
    profile_id          VARCHAR,
    search_keyword      VARCHAR,
    job_title           VARCHAR,
    rule_verdict        VARCHAR NOT NULL,   -- accept | review | reject
    rule_reason         VARCHAR,
    llm_is_relevant     BOOLEAN,            -- set by enricher (Agent 2)
    llm_confidence      DOUBLE,             -- 0-1
    llm_reason          VARCHAR,
    ensemble_verdict    VARCHAR NOT NULL,   -- accept | review | reject (derived)
    judged_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_judged_profile   ON judged_job_offers(profile_id);
CREATE INDEX IF NOT EXISTS idx_judged_keyword   ON judged_job_offers(search_keyword);
CREATE INDEX IF NOT EXISTS idx_judged_ensemble  ON judged_job_offers(ensemble_verdict);

-- Idempotent migrations (DuckDB does not support constraints in ALTER TABLE).
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS job_id            VARCHAR;
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS profile_id        VARCHAR;
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS search_keyword    VARCHAR;
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS job_title         VARCHAR;
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS rule_verdict      VARCHAR;
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS rule_reason       VARCHAR;
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS llm_is_relevant   BOOLEAN;
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS llm_confidence    DOUBLE;
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS llm_reason        VARCHAR;
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS ensemble_verdict  VARCHAR;
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS judged_at         TIMESTAMP;

-- Idempotent migration: add bi-encoder score column.
ALTER TABLE judged_job_offers ADD COLUMN IF NOT EXISTS biencoder_score DOUBLE;
