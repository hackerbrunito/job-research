-- Full-refresh mart_salary_by_keyword with p25/p50/p75 percentiles
-- on min_amount and max_amount.
DELETE FROM mart_salary_by_keyword;

INSERT INTO mart_salary_by_keyword (
    search_keyword, currency, period,
    p25_min, p50_min, p75_min,
    p25_max, p50_max, p75_max,
    observation_count, refreshed_at
)
SELECT
    f.search_keyword                                   AS search_keyword,
    COALESCE(s.currency, '')                           AS currency,
    COALESCE(s.period, '')                             AS period,
    percentile_cont(0.25) WITHIN GROUP (ORDER BY s.min_amount) AS p25_min,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY s.min_amount) AS p50_min,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY s.min_amount) AS p75_min,
    percentile_cont(0.25) WITHIN GROUP (ORDER BY s.max_amount) AS p25_max,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY s.max_amount) AS p50_max,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY s.max_amount) AS p75_max,
    COUNT(*)                                           AS observation_count,
    CURRENT_TIMESTAMP                                  AS refreshed_at
FROM fact_job_offers f
JOIN dim_salary s USING (salary_key)
WHERE s.min_amount IS NOT NULL OR s.max_amount IS NOT NULL
GROUP BY f.search_keyword, COALESCE(s.currency, ''), COALESCE(s.period, '');
