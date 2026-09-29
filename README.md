# Clinical Trial Risk Pipeline

Ranks Phase 1–3 drug trials by their risk of early termination, from the registry record, the sponsor's track record before the trial started, and the drug's FAERS reporting history before the trial started.

**Caveat first.** Sponsor and FAERS features are computed as of each trial's start date. The registry-record features (design, eligibility, geography, text) come from the latest version of each record, because the AACT snapshot holds only that version. Records are edited during trials, so the held-out AUC below may be inflated by post-start edits. M6 measures this (`docs/plans/2026-09-29-m6-honest-model.md`). Until then, treat the results as preliminary: AUC measures ranking, the risk percentages are calibrated only on the 2015–2016 cohort, and per-trial contributors are model contributions, not causes.

Design: `docs/specs/2026-09-26-design.md`.

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
make dashboard                                # Snowflake serving views -> docs/dashboard/index.html
make audit-build                              # M6: registry features from archived AACT snapshots (AACT_ARCHIVES in .env)
make audit                                    # M6: latest-record vs point-in-time AUC on 2017-2020 starts
make backtest                                 # scores written earlier vs outcomes known now
make report                                   # docs/model_card.md: the full validation report for the latest model
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

## Results (model v2, trained on starts 2008–2014, tested on resolved 2015–2016)

| Target | Test ROC AUC (95% CI) | Baseline (logistic) | Top-10% precision vs base rate |
|---|---|---|---|
| Any termination | **0.716** (0.703–0.729) | 0.678 | 33.8% vs 14.8% (2.3×) |
| Terminated for enrollment | **0.791** (0.773–0.810) | 0.764 | 17.5% vs ~5% (3.5×) |
| Terminated for safety | 0.666 (0.609–0.720) | 0.669 | 87 test positives — too few to be conclusive |

What each ingredient adds (any-termination model, test AUC): registration text +0.016, burden features +0.011, FAERS reporting history +0.002. By sponsor: industry 0.743, academic/other 0.672, government 0.671. On the censored 2017–2020 trials the AUC is 0.711, close to the headline.

### Validation summary (v2, any termination)

The full report, generated from the model's own metrics files, is [`docs/model_card.md`](docs/model_card.md): predictors and when each was measured, sample sizes, tuning, discrimination, calibration, ablations including a model without any edit-prone field, rolling-origin windows, subgroups, the point-in-time audit and the prospective backtest.

| | Train | Test | Recent |
|---|---|---|---|
| Start years | 2008–2014 | 2015–2016 | 2017–2020 |
| Trials | 36,286 | 10,242 | 19,596 |
| Terminated | ≈5,055 (13.9%) | ≈1,513 (14.8%) | ≈3,659 (18.7%) |
| Use | fit; tuned on <2013 vs 2013–2014 | headline metrics, calibration | reported with censoring caveat |

- **Outcome:** `overall_status = TERMINATED` (1) vs `COMPLETED` (0). Withdrawn, suspended, unknown-status and still-running trials carry no label and are left out of every split. The recent years over-represent early terminations, because terminated trials finish sooner.
- **Never touched by tuning or text fitting:** the 2015–2016 rows. Tuning picks from a 12-point LightGBM grid by AUC on an inner time split inside the training years, with early stopping. TF-IDF and SVD are fit on training rows only. `tests/test_ml_train.py` pins both.
- **Text model:** TF-IDF over word 1–2grams (min_df 20, at most 50k terms), TruncatedSVD to 64 components.
- **Missing data:** LightGBM's native handling; median imputation plus missing indicators for the logistic baseline.
- **Calibration (test):** Brier 0.117. Deciles in `models/v2/metrics.json` track the diagonal within 3 points; the top decile predicts 31.6% and observes 33.8%. Calibration intercept and slope, AUC by phase and by start year, and exact event counts come from the next `make train` (added in M6).
- **Main model contributors** (`top_drivers` in `metrics.json`, `TOP_DRIVER_1..3` in `TRIAL_RISK_SCORES`) are LightGBM SHAP contributions in log-odds. They say what moved a score, not why a trial would stop. From M6 on, each per-trial contributor reads `feature=value` with its signed contribution (e.g. `healthy_volunteers=No`, +0.21), and the global list gives the direction per value.

v1 (M4) scored 0.693. The v2 gain comes from registration text, trial-governance fields (DMC, responsible party, collaborators), sponsor activity, and tuning on an inner time split — after removing four leakage sources found in review (outcome measures rewritten at results posting, termination wording in summaries, negated "no safety concerns" reasons, and a sponsor feature that skewed at scoring time).

Limitations: stop reasons come from free text (safety labels are ~70–80% precise); registry features describe the latest record (see the caveat at the top); planned enrollment is excluded because the current record leaks the outcome. FAERS features are a historical reporting signal, not a measure of drug safety: FAERS has duplicate and incomplete reports and cannot establish causation or incidence (FDA). They add little here; the full 113 GB history is the next test of that.

## Dashboard

[`docs/dashboard/index.html`](docs/dashboard/index.html) is built by `make dashboard` from two Snowflake views (`VW_ACTIVE_TRIAL_RISK`, `VW_RISK_BY_AREA`) and the latest model's metrics. It shows held-out performance with confidence intervals, calibration, what each feature group adds, and the riskiest active trials with filters and their main model contributors (feature, value and signed contribution). It carries the same latest-record caveat as this README. It is one self-contained file: open it locally or serve `docs/` with GitHub Pages.

