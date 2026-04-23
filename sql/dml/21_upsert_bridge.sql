-- Upsert job_skill_bridge from a registered DataFrame `_bridge_df`.
-- DataFrame columns: job_id, skill_key.
INSERT INTO job_skill_bridge (job_id, skill_key)
SELECT job_id, skill_key
FROM _bridge_df
ON CONFLICT (job_id, skill_key) DO NOTHING;
