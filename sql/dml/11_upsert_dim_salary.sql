-- Upsert dim_salary from a registered DataFrame `_dim_salary_df`.
-- DataFrame columns: salary_key, min_amount, max_amount, currency, period.
INSERT INTO dim_salary (salary_key, min_amount, max_amount, currency, period)
SELECT salary_key, min_amount, max_amount, currency, period
FROM _dim_salary_df
ON CONFLICT (salary_key) DO NOTHING;
