-- Full-refresh mart_skills_by_keyword, keyed by (profile_id, search_keyword, skill_type, name).
DELETE FROM mart_skills_by_keyword;

INSERT INTO mart_skills_by_keyword (profile_id, search_keyword, skill_name, skill_type, demand_count, refreshed_at)
SELECT
    COALESCE(f.profile_id, '') AS profile_id,
    f.search_keyword           AS search_keyword,
    s.name                     AS skill_name,
    s.skill_type               AS skill_type,
    COUNT(*)                   AS demand_count,
    CURRENT_TIMESTAMP          AS refreshed_at
FROM fact_job_offers f
JOIN job_skill_bridge b USING (job_id)
JOIN dim_skill s USING (skill_key)
GROUP BY COALESCE(f.profile_id, ''), f.search_keyword, s.skill_type, s.name;
