-- Upsert fact_job_offers from a registered DataFrame `_fact_df`.
-- DataFrame columns: job_id, run_id, profile_id, scraped_at, enriched_at,
-- site, search_keyword, company, title, job_url, date_posted, work_mode,
-- location_key, salary_key.
INSERT INTO fact_job_offers (
    job_id, run_id, profile_id, scraped_at, enriched_at, site, search_keyword,
    company, title, job_url, date_posted, work_mode, location_key, salary_key
)
SELECT
    job_id, run_id, profile_id, scraped_at, enriched_at, site, search_keyword,
    company, title, job_url, date_posted, work_mode, location_key, salary_key
FROM _fact_df
ON CONFLICT (job_id) DO NOTHING;
