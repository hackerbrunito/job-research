-- Upsert dim_skill from a registered DataFrame `_dim_skill_df`.
-- DataFrame columns: skill_key, name, skill_type.
INSERT INTO dim_skill (skill_key, name, skill_type)
SELECT skill_key, name, skill_type
FROM _dim_skill_df
ON CONFLICT (skill_key) DO NOTHING;
