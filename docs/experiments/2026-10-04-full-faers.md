# Experiment (M10): the full FDA adverse-event history

**Question.** Until v4 the FAERS features came from a sample: the first quarter of each year 2004–2020 (279 files, 3.2 million reports). Does the drug's full safety-reporting history, every quarter from 2004 to 2026, help predict termination?

**Answer: no, it does not improve prediction.** With six times the reports, removing the FAERS features still changes test AUC by about 0.001. v5 is served anyway, because its features are complete and current where v4's were frozen at 2020, and its accuracy is equivalent.

## How the data got there

| Step | Where | Time | Cost |
|---|---|---|---|
| Copy 1,767 openFDA files (118 GB) to `gs://…/raw/faers_full/` | Cloud Run Job, 24 parallel tasks (`ingest_job/`) | 3 min 45 s | ≈ $0.10 |
| Flatten to one row per report and substance | Dataproc Serverless, 7 executors | 1 h 51 min | $3.46 |
| Match trial drugs to FDA substances | Dataproc Serverless | 2.5 min | $0.03 |

The first flatten run failed after two hours: the history contains report dates such as year 0001, which Spark refuses to write to Parquet, and its overwrite had already emptied the published table. Dates outside 1960–today are now null, and output is written to a staging folder and swapped in only when complete (`flatten_faers.publish`). The openFDA spot-check still matches exactly (ondansetron, metformin, atorvastatin, adalimumab in 2016 Q1).

| | Q1 sample (v4) | Full history (v5) |
|---|---|---|
| Reports (latest version each) | 3,247,112 | 20,685,981 |
| Report × substance rows | 11.9 M | 73.1 M |
| Substance vocabulary for matching | 7,795 | 13,707 |
| Labeled trials matched to ≥1 substance | 44,222 (66.9%) | 47,596 (72.0%) |
| Test trials with FAERS history before start | 61.8% | 64.2% |
| Median prior reports, matched test trials | 3,562 | 12,389 |

## Results (test: 10,242 trials that started 2015–2016)

| | Any termination | Enrollment termination | Safety termination |
|---|---|---|---|
| v4 (Q1 sample) | 0.703 | 0.783 | 0.665 |
| v5 (full history) | 0.708 (0.696–0.721) | 0.780 | 0.675 |
| v5 without any FAERS feature | 0.707 | 0.778 | 0.678 |

- v5 minus v4, paired bootstrap on the same trials: any termination −0.000 to +0.010; enrollment −0.011 to +0.004. Equivalent.
- Within v5, the FAERS features add +0.001 (any termination), +0.002 (enrollment) and −0.003 (safety, 87 test events). The drug's reporting history does not tell this model which trials will stop, not even for safety reasons.
- Point-in-time audit of v5 (same 25 archives, 18,568 trials): AUC 0.692 on latest records vs 0.695 on records as they stood at start (change +0.003, 95% CI 0.000 to +0.005). No inflation.
- A first v5 run picked up the M9 text embeddings by default and so changed two things at once; it is kept in `models/experiments/m10-full-faers-plus-embeddings/` and embeddings are now opt-in (`TEXT_EMBEDDINGS=1`).

## Why it is served anyway

v4's FAERS features stop at 2020 and see one quarter a year, so every trial that started after 2020 (most of today's active trials) was scored with adverse-event counts frozen in 2020. v5's come from reports through 2026, cover more trials (72% matched vs 67%), match the features now in Snowflake, and lose nothing in accuracy.

## Why FAERS does not help, probably

FAERS counts measure how much a drug is used and reported, not how risky a new trial of it is. Most terminations are for enrollment or business reasons, which a drug's post-marketing reports cannot see, and the trials where safety ends a study (under 1% here) are often of drugs with little or no marketing history, hence no FAERS reports at all.
