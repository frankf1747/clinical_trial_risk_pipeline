-- How and when each trial ended, for the survival model (M8). Kept apart from TRIAL_FEATURES on purpose:
-- durations and end dates are outcomes and must never become model inputs.
-- end_date is NULL for running trials (censored at the snapshot) and for the few finished trials whose
-- registry record has no completion date. Portable SQL: tests run it on DuckDB.

CREATE OR REPLACE VIEW TRIAL_OUTCOMES AS
SELECT f.nct_id, f.split, f.start_date, f.label, o.completion_date AS end_date
FROM TRIAL_FEATURES f
LEFT JOIN RAW_SPONSOR_OUTCOMES o ON o.nct_id = f.nct_id AND f.label IS NOT NULL;
