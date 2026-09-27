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
