-- One row per trial. Everything except `label` was knowable on the trial's start date.
-- Deliberately absent: status (it is the label), completion_date, sponsor_name.
-- start_date is kept as metadata for splitting; it is not a model input.
-- text is registration-time prose; the model turns it into features itself.
CREATE OR REPLACE TABLE TRIAL_FEATURES AS
SELECT t.nct_id,
       t.label,
       CASE WHEN t.label IS NULL THEN 'score'
            WHEN t.start_date < DATE '2015-01-01' THEN 'train'
            WHEN t.start_date < DATE '2017-01-01' THEN 'test'      -- resolved years: headline metric
            ELSE 'recent' END                                      AS split,  -- 2017-2020: still censored
       -- Reason-specific targets: 1 = terminated for that reason, 0 = completed,
       -- NULL = terminated for another reason (left out of that model). stop_reason itself stays out.
       CASE WHEN t.label = 0 THEN 0 WHEN t.stop_reason = 'enrollment' THEN 1 END AS label_enrollment,
       CASE WHEN t.label = 0 THEN 0 WHEN t.stop_reason = 'safety'     THEN 1 END AS label_safety,
       t.start_date,
       t.phase, t.number_of_arms, t.allocation, t.intervention_model, t.primary_purpose,
       t.masking, t.sponsor_class, t.has_dmc,
       a.* EXCLUDE (nct_id),
       COALESCE(sh.sponsor_prior_trials, 0)                        AS sponsor_prior_trials,
       sh.sponsor_prior_termination_rate,
       COALESCE(sc.sponsor_trials_started_2y, 0)                   AS sponsor_trials_started_2y,
       COALESCE(ds.n_substances, 0)                                AS n_substances,
       COALESCE(ds.faers_reports, 0)                               AS faers_reports,
       COALESCE(ds.faers_reports_12m, 0)                           AS faers_reports_12m,
       ds.faers_serious_reports::FLOAT / NULLIF(ds.faers_reports, 0) AS faers_serious_share,
       ds.faers_death_reports::FLOAT / NULLIF(ds.faers_reports, 0)   AS faers_death_share,
       COALESCE(ds.faers_reports, 0) > 0                           AS has_faers_history,
       x.text
FROM RAW_TRIALS t
LEFT JOIN RAW_TRIAL_ATTRIBUTES a ON a.nct_id = t.nct_id
LEFT JOIN INT_SPONSOR_HISTORY sh ON sh.nct_id = t.nct_id
LEFT JOIN INT_SPONSOR_RECENT_STARTS sc ON sc.nct_id = t.nct_id
LEFT JOIN INT_DRUG_SAFETY ds     ON ds.nct_id = t.nct_id
LEFT JOIN RAW_TRIAL_TEXT x       ON x.nct_id = t.nct_id;
