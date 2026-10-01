# Point-in-time audit, v3 — partial (2017 archives)

**Status: preliminary.** The audit is meant to read all 49 monthly AACT archives from 2017-01 to 2021-01. This run used the 12 archives from 2017; the 37 archives for 2018–2021 are not downloaded yet. So no decision is made here (plan Task 7 Step 5). The numbers below are a first reading, on the trials whose records 2017 archives can supply.

Model: v3 (`models/v3/`). Raw output: `models/v3/audit_point_in_time.json`. Method: `src/ctrisk/spark/point_in_time.py` (each trial's last archived record on or before its start, else its first record after) and `src/ctrisk/ml/audit.py` (score the same trials twice, on latest and on archived registry fields; FAERS and sponsor-history features stay as they are, being point-in-time already).

## Archives

AACT moved its downloads behind sign-in (2026-09). Archives are listed under Snapshot history → Flat Text Files, one per month, and each download link redirects to a signed storage URL. The listing date is not in the file; each zip was named by the latest `last_changed_date` in its `studies.txt`, which matches the listing date to within a few days. (`nlm_download_date_description` is per study, not per snapshot, and cannot date an archive.)

| Archive (max last change) | AACT listing | Cohort trials present |
|---|---|---|
| 2017-01-03 | Jan 5 | 1,246 |
| 2017-02-16 | Feb 18 | 1,710 |
| 2017-03-07 | Mar 9 | 1,797 |
| 2017-03-30 | Apr 3 | 1,965 |
| 2017-04-27 | May 2 | 2,472 |
| 2017-06-08 | Jun 13 | 3,027 |
| 2017-06-28 | Jul 3 | 3,342 |
| 2017-08-09 | Aug 11 | 3,831 |
| 2017-08-31 | Sep 6 | 4,186 |
| 2017-10-13 | Oct 17 | 4,655 |
| 2017-10-31 | Nov 2 | 4,882 |
| 2017-12-14 | Dec 17 | 5,482 |

5,482 of the 19,596 labeled 2017–2020 starts have a record in these archives. Lag from record to start: median −9 days (p10 −103, p90 +41); 1,694 were first seen after they started (registered late).

## Format drift found and fixed before reading any result

Each was a change in how AACT wrote a field, not an edit to a record. Each fix is tested; rebuilding today's snapshot with the fixes changes 0 of 111,118 trial rows and 0 attribute rows, so v3's inputs are untouched.

| Field | 2017 archives | Fix |
|---|---|---|
| start date | no `start_date`; `start_month_year` ("January 2015") | parse it; month-only dated to the month's last day, as AACT does now |
| masking (Jan 2017) | "Double Blind" for any 2–4 masked parties | level from the masked-party columns (4 parties → QUADRUPLE) |
| masking (Feb–Aug 2017) | the parties themselves ("Participant, Investigator"), "No masking" | count the named parties; "No masking" → NONE |
| allocation | blank for single-arm trials | → NA, as the current registry says |
| sponsor class | "U.S. Fed" | → GOVERNMENT |
| sex | "Both" | → ALL |
| primary purpose | "Educational/Counseling/Training" | → ECT |
| eligibility criteria | hard-wrapped at ~80 characters, separated by `~` or only by runs of spaces | rebuilt one line per criterion with `* ` bullets; criterion counts then match today's for ~3 in 4 trials (from 1–18%) |
| disease areas | no MeSH ancestors listed, so top-level areas read False for nearly all trials | marked unknown; the audit keeps the latest value (all 14 `area_*` columns, 100% of trials here) |

Before the masking and criteria fixes, 38% of archived masking values were categories the model never saw and `criteria_count` "changed" for 99% of trials in both groups. After them, the remaining unseen levels are `phase` 2.2% (records whose phase was later edited from N/A, Phase 4 or Early Phase 1 into the modelled Phase 1–3; genuine edits) and `primary_purpose` 0.04%.

## Result (preliminary)

| Trials | n | Terminated | Latest-record AUC (95% CI) | Point-in-time AUC (95% CI) | Drop (paired 95% CI) |
|---|---|---|---|---|---|
| with an archived record | 5,482 | 1,071 | 0.676 (0.658–0.693) | 0.668 (0.651–0.685) | 0.008 (0.002 to 0.015) |
| record predates start | 3,788 | 839 | 0.644 (0.624–0.665) | 0.638 (0.618–0.658) | 0.007 (−0.002 to 0.016) |

Calibration (intercept, slope) on the matched trials: latest 0.34, 0.94; point-in-time 0.36, 0.89. The positive intercept is the censoring of recent starts (terminated trials finish first, so the resolved 2017–2020 trials over-represent terminations), not the swap.

The matched trials score lower than the whole 2017–2020 cohort (0.676 vs 0.713 on latest-record features) because they are a different population: trials already registered by December 2017, so mostly 2017 starts and long-planned later ones.

### Which fields carry it

AUC drop when only that column is swapped to its archived value, and how often it changed between the archived record and now:

| Field | Changed, terminated | Changed, completed | Gap (points) | AUC drop, this column alone |
|---|---|---|---|---|
| `criteria_count` | 54.5% | 46.4% | +8.1 | 0.000 |
| `n_keywords` | 12.2% | 8.1% | +4.2 | 0.000 |
| `criteria_chars` | 88.8% | 85.0% | +3.8 | 0.000 |
| `n_countries` | 41.3% | 38.0% | +3.3 | 0.002 |
| `us_only` | 17.3% | 14.0% | +3.2 | 0.007 |
| `phase` | 8.0% | 5.4% | +2.7 | 0.001 |
| `responsible_party` | 7.6% | 5.0% | +2.5 | 0.001 |
| `number_of_arms` | 13.6% | 11.6% | +2.0 | 0.000 |
| `n_collaborators` | 11.1% | 10.0% | +1.1 | 0.001 |
| `text` | 100% | 100% | 0 | −0.003 |

- Almost all of the drop is geography: `us_only` (0.007) and `n_countries` (0.002). Terminated trials' country lists changed after start more often than completed trials' (us_only 17.3% vs 14.0%), and the latest record carries that change back to the start. (Which way they changed is not broken out here.)
- `criteria_count` is the only field over the plan's 5-point gap threshold, but swapping it moves AUC by 0.000, so the model is not leaning on its post-start values. Part of its 46–55% change rate is residual formatting the unwrap does not catch; `criteria_chars` (an exact character count) is dominated by that residue.
- The text changes for every trial (format), and its archived version scores slightly *better* (−0.003): no sign of text leakage.

## Against the decision rule (plan Task 7 Step 5) — not applied yet

- Drop ≤ 0.01: met on these trials (0.008; strict subset 0.007).
- No field with a terminated-minus-completed change gap above 5 points: not met, on `criteria_count` alone (+8.1), which carries no AUC.

If the full set looks like this, the reading is the first outcome with one qualification: keep v3 and the latest-record caveat, report the drop, and consider building `us_only`/`n_countries` from the first registered country list (the leak, though small, sits there). The rule cannot be applied on 5,482 trials that skew to 2017 starts: the 2018–2020 starts, whose records spent longer between registration and today, are where edits would accumulate.

## To finish

1. Allow multiple downloads for aact.ctti-clinicaltrials.org in Chrome; download the 37 flat-file archives 2018-01 … 2021-01 (sorted by date with the same rule).
2. `make audit-build AACT_ARCHIVES="data/raw/aact_archive_zips/*.zip"` (2017 archives are skipped as already reduced), check the per-archive table for new drift (the 2019–2020 exports may change format again), then `make audit report`.
3. Replace this note with the full one and apply the decision rule.
