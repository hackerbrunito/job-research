-- Full-refresh mart_jobs_by_country.
DELETE FROM mart_jobs_by_country;

INSERT INTO mart_jobs_by_country (country_code, country, search_keyword, job_count, refreshed_at)
SELECT
    COALESCE(l.country_code, '') AS country_code,
    MAX(l.country)               AS country,
    f.search_keyword             AS search_keyword,
    COUNT(*)                     AS job_count,
    CURRENT_TIMESTAMP            AS refreshed_at
FROM fact_job_offers f
LEFT JOIN dim_location l USING (location_key)
GROUP BY COALESCE(l.country_code, ''), f.search_keyword;
