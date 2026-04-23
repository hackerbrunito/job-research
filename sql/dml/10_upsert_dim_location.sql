-- Upsert dim_location from a registered DataFrame `_dim_location_df`.
-- DataFrame columns: location_key, city, country, country_code.
INSERT INTO dim_location (location_key, city, country, country_code)
SELECT location_key, city, country, country_code
FROM _dim_location_df
ON CONFLICT (location_key) DO NOTHING;
