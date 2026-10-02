# M7 design: a public trial lookup, and each platform doing real work

**Goal.** Anyone with a link can look up a Phase 1–3 drug trial by NCT ID and see its termination-risk score, how it ranks against active trials, and the main reasons, in plain English. Each platform carries a real part of the pipeline: GCS is the data lake, Dataproc Serverless runs the Spark jobs, Snowflake builds features and holds every score, Cloud Run serves the lookup. No scheduled runs; each refresh is started by hand with `make`.

## Decisions

| Question | Choice | Why |
|---|---|---|
| Who uses it | Anyone, by NCT ID | A hiring reviewer or analyst can try the model without an account |
| Access and cost | Public Cloud Run service reading a precomputed file | Scales to zero; no Snowflake credentials or warehouse time per visitor |
| Coverage | Active and finished trials in the modelled population (~111,000) | Finished trials let a reader sanity-check the model |
| Page content | Score, percentile, enrollment risk, top reasons in plain English, trial status | Kept to what the score means; no browse or peer pages in v1 |
| Serving pattern | Batch scores → Snowflake → unload to GCS → app | Alternatives (Snowpark UDF, live inference) add cost or packaging with no user-visible gain |
| Spark | Dataproc Serverless, same modules, `MODE=cloud` | Makes the full FAERS history feasible later; local mode stays for tests |

## Honest scores for every trial

A score must never come from a model that saw that trial's outcome.

| Trials | Score | Label in the app |
|---|---|---|
| Active (`split = score`) | the version's final model | forward-looking |
| Finished, started 2015+ (`test`, `recent`, `later`) | the final model, trained on 2008–2014 only | held out |
| Finished, started 2008–2014 (`train`) | leave-one-start-year-out: seven models with the version's parameters, each trained without that year, text refit per fold | out of fold |

Percentile: the share of active trials scored by the same version with a lower score, for every trial, so a finished trial reads on the same scale. Reasons: the three contributions pushing risk up most and the two pushing it down most, from the model that produced the score, rendered in plain English ("Does not accept healthy volunteers: raises risk"). Contributions describe the model, not causes; the page says so.

Scores go to a new table, `TRIAL_LOOKUP_SCORES` (one row per trial per version, replaced on rerun). `TRIAL_RISK_SCORES` stays as it is: active trials only, append-only, the basis of the prospective backtest, which must never see held-out or out-of-fold rows.

## Data flow

```
raw AACT tables, FAERS zips ──► gs://BUCKET/raw/
Dataproc Serverless batches (trials, match, attributes, text, faers) ──► gs://BUCKET/parquet/
Snowflake: stage → RAW_* → INT_* → TRIAL_FEATURES (as-of SQL) → checks → TRIAL_FEATURES_Vn clones
local Python: train vN; score active (TRIAL_RISK_SCORES); score all (TRIAL_LOOKUP_SCORES)
Snowflake: TRIAL_LOOKUP view → COPY INTO @SERVING_STAGE ──► gs://BUCKET/serving/lookup_vN.parquet
local Python: gs://BUCKET/serving/current.json (version, file, model summary)
Cloud Run: reads current.json and the lookup file at start; serves / , /trial/{nct_id}, /api/trials/{nct_id}
```

## Components

- `src/ctrisk/ml/lookup.py`: `cross_fit` and `score_all` (pure, tested on synthetic data); `__main__` writes `TRIAL_LOOKUP_SCORES`.
- `src/ctrisk/serving/labels.py`: plain-English labels and value formatting for every model feature.
- `sql/40_lookup.sql`: `TRIAL_LOOKUP` view (latest version per trial, with title, sponsor, phase, status, start date) and the unload. Portable parts run on DuckDB in tests.
- `src/ctrisk/serving/publish.py`: runs the unload and writes `current.json`. `make publish`.
- `lookup_app/`: FastAPI app, Dockerfile, its own small requirements; tested with FastAPI's test client on a fixture file.
- `src/ctrisk/cloud/dataproc.py`: builds the wheel, uploads code, submits a module as a Dataproc Serverless batch with a size cap and TTL; `--dry-run` prints the command and a cost estimate. `make <job> MODE=cloud` routes through it.
- `scripts/gcp_setup.sh` and `sql/01_serving_setup.sql`: the one-time permission steps the account owner runs (APIs, IAM, storage integration). They change account security settings, so they are not run by the pipeline.

## Errors and guardrails

- Unknown NCT ID → 404 page that says the trial is outside the modelled population (Phase 1–3 drug trials started 2008+) or not in the snapshot.
- The app refuses to start if `current.json` or the lookup file is missing or has no rows.
- Every Dataproc batch carries a TTL and an executor cap; the dispatcher prints the estimated cost before submitting.

## Testing

Unit tests for cross-fitting (no trial scored by a model that saw it), percentiles, reasons and labels; DuckDB test for the lookup view; app tests for found, not found and health; dispatcher test for the built command (no cloud calls).
