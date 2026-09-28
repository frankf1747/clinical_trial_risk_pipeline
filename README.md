# Clinical Trial Risk Pipeline

Predicts which drug trials will be terminated early, using only what is known when they start.

Work in progress — see `docs/specs/2026-09-26-design.md`.

## Run

```bash
make setup
make ingest-aact AACT=<aact-flat-files.zip>   # from https://aact.ctti-clinicaltrials.org/snapshots
make ingest-faers                             # FAERS sample set in .env (default: Q1 2004-2020, 14.9 GB)
make trials                                   # -> data/parquet/trials, drug_interventions
make faers                                    # -> data/parquet/faers_drug_events
make match                                    # -> data/parquet/trial_drug_map
make check-faers                              # our counts vs the openFDA API
make attributes                               # -> data/parquet/trial_attributes
make text                                     # -> data/parquet/trial_text
make upload                                   # data/parquet -> gs://$GCP_BUCKET/parquet
make warehouse                                # Snowflake: RAW_* -> TRIAL_FEATURES, then checks
make train                                    # clone TRIAL_FEATURES, fit, save models/vN
make score                                    # overall + enrollment risk for active trials -> TRIAL_RISK_SCORES
make test
```

## Snowflake setup (once)

```bash
mkdir -p ~/.snowflake && openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out ~/.snowflake/rsa_key.p8 -nocrypt && chmod 600 ~/.snowflake/rsa_key.p8
openssl rsa -in ~/.snowflake/rsa_key.p8 -pubout | grep -v -- '-----' | tr -d '\n'
```

In Snowsight: `ALTER USER <your user> SET RSA_PUBLIC_KEY='<the printed key>';`

Then run `sql/00_setup.sql` in Snowsight (as `ACCOUNTADMIN`, top to bottom), copy `STORAGE_GCP_SERVICE_ACCOUNT` from its `DESC STORAGE INTEGRATION` output, and grant that service account `roles/storage.objectViewer` on the bucket:

```bash
gcloud storage buckets add-iam-policy-binding gs://$GCP_BUCKET \
  --member="serviceAccount:<STORAGE_GCP_SERVICE_ACCOUNT>" --role=roles/storage.objectViewer
```

Fill in the Snowflake block in `.env` (see `.env.example`).

## Status

M1 on the AACT snapshot of 2026-09-26 (604,561 registered studies):

| | Trials |
|---|---|
| Labeled (Phase 1–3 drug trials, started 2008–2020, completed or terminated) | 66,124 |
| Termination rate | 15.5% |
| Active (to be scored) | 29,894 |
| Drug/biologic interventions to match against FAERS | 208,568 (80,199 distinct names) |

M2 on the FAERS sample (Q1 of each year 2004–2020; 279 files, 14.9 GB zipped, ~83 GB JSON):

| | |
|---|---|
| Reports (latest version each) | 3,247,112 of 3,248,203 listed by openFDA |
| Report × substance rows | 11.9M (flatten: ~2 min on a laptop) |
| Match vocabulary | 7,795 FDA-coded substances (openFDA standard + FAERS active-substance names, minus vague/inert) |
| Labeled trials matched to ≥1 substance | 44,222 of 66,124 (66.9%) |
| API spot-check (2016 Q1, ours vs openFDA) | ondansetron, metformin, atorvastatin, adalimumab: all 1.00 |
| Random sample of 30 matches | 30 correct |

Unmatched trials are mostly investigational compounds with no marketing history (e.g. "BI 10773"), drugs never sold in the US, placebos, and non-drug interventions.

M3 in Snowflake (`TRIAL_FEATURES`, built from GCS in ~50 s; all checks pass):

| Split | Trials | Termination rate | With FAERS history before start |
|---|---|---|---|
| train (started 2008–2016) | 46,528 | 14.1% | 62.1% |
| test (started 2017–2020) | 19,596 | 18.7% | 58.4% |
| score (active) | 29,894 | — | 57.1% |

The test split's higher termination rate is expected: only trials finished by the snapshot have a label, and terminated trials finish sooner. A hand check of one trial (NCT00456846) against the raw tables matched on every FAERS and sponsor feature.

