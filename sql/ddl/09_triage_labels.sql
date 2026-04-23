-- User-curated per-profile title labels.
-- One row per (profile_id, title_norm) pair.
-- Future scrapes check this table to pre-block known bad patterns.
CREATE TABLE IF NOT EXISTS profile_title_labels (
    profile_id   VARCHAR NOT NULL,
    title_norm   VARCHAR NOT NULL,        -- lowercase, stripped title used as lookup key
    label        VARCHAR NOT NULL,        -- 'accept' | 'reject' | 'unsure'
    note         VARCHAR,                 -- optional user note
    count_seen   INTEGER NOT NULL DEFAULT 1,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY  (profile_id, title_norm)
);
