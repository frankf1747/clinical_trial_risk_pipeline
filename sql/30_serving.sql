-- Views for the people who use the scores. Built by `make dashboard`, after `make score`.
-- Portable SQL: tests run it on DuckDB.

-- Scores written before M6 have no contribution columns; add them so the view reads every version.
ALTER TABLE TRIAL_RISK_SCORES ADD COLUMN IF NOT EXISTS top_driver_1_contrib FLOAT;
ALTER TABLE TRIAL_RISK_SCORES ADD COLUMN IF NOT EXISTS top_driver_2_contrib FLOAT;
ALTER TABLE TRIAL_RISK_SCORES ADD COLUMN IF NOT EXISTS top_driver_3_contrib FLOAT;

-- Each active trial with its most recent score and what a reviewer needs to recognize it.
-- Disease area is the first match in a fixed order, so every trial lands in exactly one.
CREATE OR REPLACE VIEW VW_ACTIVE_TRIAL_RISK AS
WITH latest AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY nct_id ORDER BY scored_at DESC, model_version DESC) AS rn
    FROM TRIAL_RISK_SCORES
)
SELECT l.nct_id, t.brief_title, t.phase, t.sponsor_class, f.start_date,
       CASE WHEN f.area_neoplasms THEN 'Oncology'
            WHEN f.area_cardiovascular THEN 'Cardiovascular'
            WHEN f.area_nervous_system THEN 'Neurology'
            WHEN f.area_mental THEN 'Psychiatry'
            WHEN f.area_infections THEN 'Infectious disease'
            WHEN f.area_respiratory THEN 'Respiratory'
            WHEN f.area_metabolic THEN 'Metabolic'
            WHEN f.area_endocrine THEN 'Endocrine'
            WHEN f.area_immune THEN 'Immunology'
            WHEN f.area_digestive THEN 'Digestive'
            WHEN f.area_skin THEN 'Dermatology'
            WHEN f.area_musculoskeletal THEN 'Musculoskeletal'
            WHEN f.area_urogenital THEN 'Urogenital'
            WHEN f.area_blood THEN 'Hematology'
            ELSE 'Other' END                              AS disease_area,
       l.risk_score, l.risk_decile, l.enrollment_risk_score,
       l.top_driver_1, l.top_driver_2, l.top_driver_3,
       l.top_driver_1_contrib, l.top_driver_2_contrib, l.top_driver_3_contrib, l.model_version, l.scored_at
FROM latest l
JOIN TRIAL_FEATURES f ON f.nct_id = l.nct_id AND f.split = 'score'
JOIN RAW_TRIALS t     ON t.nct_id = l.nct_id
WHERE l.rn = 1;

CREATE OR REPLACE VIEW VW_RISK_BY_AREA AS
SELECT disease_area,
       COUNT(*)                     AS trials,
       AVG(risk_score)              AS avg_risk,
       AVG(enrollment_risk_score)   AS avg_enrollment_risk,
       COUNT_IF(risk_decile = 10)   AS top_decile_trials
FROM VW_ACTIVE_TRIAL_RISK
GROUP BY disease_area;
