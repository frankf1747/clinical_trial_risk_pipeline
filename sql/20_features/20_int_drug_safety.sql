-- FAERS history of each trial's drugs, counting only months that ended before the trial's
-- start month. The start month itself is excluded, so no report dated on or after the start
-- can leak in. A report naming two of a trial's drugs counts once per drug.
CREATE OR REPLACE TABLE INT_DRUG_SAFETY AS
SELECT t.nct_id,
       COUNT(DISTINCT f.substance)            AS n_substances,   -- only drugs with history before start
       COALESCE(SUM(f.reports), 0)            AS faers_reports,
       COALESCE(SUM(f.serious_reports), 0)    AS faers_serious_reports,
       COALESCE(SUM(f.death_reports), 0)      AS faers_death_reports,
       COALESCE(SUM(CASE WHEN f.month >= DATE_TRUNC('month', t.start_date) - INTERVAL '12 months'
                         THEN f.reports END), 0) AS faers_reports_12m
FROM RAW_TRIALS t
JOIN RAW_TRIAL_DRUG_MAP m ON m.nct_id = t.nct_id
LEFT JOIN INT_FAERS_MONTHLY f
       ON f.substance = m.substance
      AND f.month < DATE_TRUNC('month', t.start_date)
GROUP BY t.nct_id;
