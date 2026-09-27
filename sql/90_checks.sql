-- check: one row per trial
SELECT nct_id FROM TRIAL_FEATURES GROUP BY nct_id HAVING COUNT(*) > 1;

-- check: every trial has a feature row
SELECT nct_id FROM RAW_TRIALS WHERE nct_id NOT IN (SELECT nct_id FROM TRIAL_FEATURES);

-- check: labels only in train and test
SELECT nct_id FROM TRIAL_FEATURES WHERE (label IS NULL) <> (split = 'score');

-- check: test split is not empty
SELECT COUNT(*) FROM TRIAL_FEATURES WHERE split = 'test' HAVING COUNT(*) = 0;

-- check: shares are proportions
SELECT nct_id FROM TRIAL_FEATURES
WHERE faers_serious_share NOT BETWEEN 0 AND 1 OR faers_death_share NOT BETWEEN 0 AND 1;

-- check: every RAW table loaded
SELECT 'RAW_TRIALS' FROM RAW_TRIALS HAVING COUNT(*) = 0
UNION ALL SELECT 'RAW_TRIAL_ATTRIBUTES' FROM RAW_TRIAL_ATTRIBUTES HAVING COUNT(*) = 0
UNION ALL SELECT 'RAW_SPONSOR_OUTCOMES' FROM RAW_SPONSOR_OUTCOMES HAVING COUNT(*) = 0
UNION ALL SELECT 'RAW_TRIAL_DRUG_MAP' FROM RAW_TRIAL_DRUG_MAP HAVING COUNT(*) = 0
UNION ALL SELECT 'RAW_FAERS_DRUG_EVENTS' FROM RAW_FAERS_DRUG_EVENTS HAVING COUNT(*) = 0;

-- check: sponsor outcomes are unique per study (stale files on GCS would duplicate them)
SELECT nct_id FROM RAW_SPONSOR_OUTCOMES GROUP BY nct_id HAVING COUNT(*) > 1;

-- check: FAERS rows are unique per report and substance
SELECT safetyreportid, substance FROM RAW_FAERS_DRUG_EVENTS
GROUP BY safetyreportid, substance HAVING COUNT(*) > 1;
