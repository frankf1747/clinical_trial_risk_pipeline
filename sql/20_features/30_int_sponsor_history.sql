-- The lead sponsor's track record: its studies that finished before this trial started.
CREATE OR REPLACE TABLE INT_SPONSOR_HISTORY AS
SELECT t.nct_id,
       COUNT(s.nct_id)                    AS sponsor_prior_trials,
       AVG(s.terminated::FLOAT)           AS sponsor_prior_termination_rate
FROM RAW_TRIALS t
LEFT JOIN RAW_SPONSOR_OUTCOMES s
       ON s.sponsor_name = t.sponsor_name
      AND s.completion_date < t.start_date
      AND s.nct_id <> t.nct_id
GROUP BY t.nct_id;

-- The sponsor's other studies (any status, not withdrawn) started in the 24 months before this
-- trial started. RAW_SPONSOR_OUTCOMES only has finished studies, so counting "concurrent" trials
-- from it undercounts at serving time (many of the sponsor's still-running studies aren't in
-- there yet) relative to training time (where they'd since finished) -- a train/serve skew.
-- RAW_SPONSOR_STARTS has every started study regardless of status, so a start date is knowable
-- at serving time exactly as it was at training time.
CREATE OR REPLACE TABLE INT_SPONSOR_RECENT_STARTS AS
SELECT t.nct_id, COUNT(s.nct_id) AS sponsor_trials_started_2y
FROM RAW_TRIALS t
LEFT JOIN RAW_SPONSOR_STARTS s
       ON s.sponsor_name = t.sponsor_name
      AND s.nct_id <> t.nct_id
      AND s.start_date >= t.start_date - INTERVAL '24 months'
      AND s.start_date < t.start_date
GROUP BY t.nct_id;
