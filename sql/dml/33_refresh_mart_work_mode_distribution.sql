-- Full-refresh mart_work_mode_distribution, keyed by (profile_id, search_keyword, work_mode).
DELETE FROM mart_work_mode_distribution;

INSERT INTO mart_work_mode_distribution (profile_id, search_keyword, work_mode, job_count, refreshed_at)
SELECT
    COALESCE(f.profile_id, '') AS profile_id,
    f.search_keyword           AS search_keyword,
    COALESCE(f.work_mode, '')  AS work_mode,
    COUNT(*)                   AS job_count,
    CURRENT_TIMESTAMP          AS refreshed_at
FROM fact_job_offers f
GROUP BY COALESCE(f.profile_id, ''), f.search_keyword, COALESCE(f.work_mode, '');
