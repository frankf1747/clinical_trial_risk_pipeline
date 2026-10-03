-- The public lookup's table: every trial's score from the newest model version, with what a reader needs
-- to recognize the trial. Built by `make publish`, after `make lookup`. Portable SQL: tests run it on DuckDB.
-- The unload to GCS (COPY INTO @SERVING_STAGE) is Snowflake-only and lives in ctrisk.serving.publish.

CREATE OR REPLACE VIEW TRIAL_LOOKUP AS
WITH latest AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY nct_id
                                 ORDER BY CAST(SUBSTR(model_version, 2) AS INT) DESC) AS rn
    FROM TRIAL_LOOKUP_SCORES
)
SELECT l.nct_id, t.brief_title, t.phase, t.sponsor_name, t.sponsor_class, t.status, t.start_date,
       l.model_version, l.score_type, l.risk_score, l.risk_percentile, l.enrollment_risk_score, l.reasons,
       l.years_running, l.next_2y_risk, l.next_2y_percentile
FROM latest l
JOIN RAW_TRIALS t ON t.nct_id = l.nct_id
WHERE l.rn = 1;
