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
make test
```

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

