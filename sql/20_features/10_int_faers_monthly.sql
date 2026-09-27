-- Reports per substance per month: 12M report rows -> ~0.5M, so the per-trial join stays small.
CREATE OR REPLACE TABLE INT_FAERS_MONTHLY AS
SELECT substance,
       DATE_TRUNC('month', receivedate) AS month,
       COUNT(*)            AS reports,
       COUNT_IF(serious)   AS serious_reports,
       COUNT_IF(death)     AS death_reports
FROM RAW_FAERS_DRUG_EVENTS
WHERE receivedate IS NOT NULL
GROUP BY substance, DATE_TRUNC('month', receivedate);
