# Point-in-time audit, v3

**Result:** taking registry fields as they stood at each trial's start, instead of from today's record, lowers v3's AUC on 2017–2020 starts by **0.005** (paired 95% CI 0.002–0.009; 18,568 trials). On the 9,112 trials whose archived record predates the start, the drop is 0.006 (0.001–0.012). Four fields carry it: `criteria_count`, `n_countries`, `criteria_chars` and `us_only` changed after start more often for trials that ended up terminated, and swapping just those four reproduces the whole drop. **Decision (plan Task 7 Step 5): the second outcome.** The drop is under 0.01, but specific fields show label-dependent change above 5 points, so v4 should remove them (they cannot be made point-in-time for the 2008–2016 training years), then be re-audited.

Model: v3 (`models/v3/`). Raw output: `models/v3/audit_point_in_time.json`. Method: `src/ctrisk/spark/point_in_time.py` (each trial's last archived record on or before its start, else its first record after) and `src/ctrisk/ml/audit.py` (score the same trials twice, on latest and on archived registry fields; FAERS and sponsor-history features stay as they are, being point-in-time already).

## Archives

25 AACT flat-file archives: every month of 2017, then one per quarter from January 2018 to January 2021. AACT serves them after sign-in under Snapshot history → Flat Text Files; each download redirects to a signed storage URL. The listing date is not in the file, so each zip was named by the latest record change in its `studies.txt` (`last_changed_date` through early 2018, `last_update_submitted_date` after), which falls within a few days before the listing date. (`nlm_download_date_description` is per study, not per snapshot, and cannot date an archive.)

| Archives | AACT listings | Cohort trials present by the last one |
|---|---|---|
| 2017-01-03 … 2017-12-14 (12) | monthly, Jan 5 … Dec 17, 2017 | 5,482 |
| 2018-01-09, 03-29, 06-28, 09-27 | Jan 11, Apr 1, Jul 1, Oct 1, 2018 | 9,141 |
| 2018-12-28, 2019-03-28, 06-27, 09-27 | Jan 1, Apr 1, Jul 1, Oct 1, 2019 | 13,561 |
| 2019-12-27, 2020-03-30, 06-29, 10-01 | Jan 1, Apr 1, Jul 1, Oct 1, 2020 | 17,833 |
| 2020-12-30 | Jan 1, 2021 | 18,568 |

18,568 of the 19,596 labeled 2017–2020 starts have an archived record; the other 1,028 were registered after December 2020. Record date minus start date: median +2 days (p10 −54, p90 +108). 9,456 trials were first seen after they started, mostly because quarterly archives miss trials registered in the weeks before their start; the strict subset below leaves them out.

## Format drift found and fixed before reading any result

Each was a change in how AACT wrote a field, not an edit to a record. Each fix is tested; rebuilding today's snapshot with the fixes changes 0 of 111,118 trial rows and 0 attribute rows, so v3's inputs are untouched. The per-archive null shares and category mixes then move smoothly across all 25 archives, with no step changes.

| Field | Archives | Fix |
|---|---|---|
| start date | 2017 to early 2018: no `start_date`, only `start_month_year` ("January 2015") | parse it; month-only dated to the month's last day, as AACT does now |
| masking | Jan 2017: "Double Blind" for any 2–4 masked parties | level from the masked-party columns (4 parties → QUADRUPLE) |
| masking | Feb–Aug 2017: the parties themselves ("Participant, Investigator"), "No masking" | count the named parties; "No masking" → NONE |
| allocation | 2017: blank for single-arm trials | → NA, as the registry says from late 2017 |
| sponsor class | "U.S. Fed" | → GOVERNMENT |
| sex | "Both" | → ALL |
| primary purpose | "Educational/Counseling/Training" | → ECT |
| eligibility criteria | all archives: hard-wrapped at ~80 characters, separated by `~` or only by runs of spaces | rebuilt one line per criterion with `* ` bullets; criterion counts then match today's for ~3 in 4 trials (from 1–18%) |
| disease areas | all archives: no MeSH ancestors listed, so top-level areas read False for nearly every trial | marked unknown; the audit keeps the latest value (all 14 `area_*` columns, 100% of trials) |

Remaining categories the model never saw: `phase` 1.2% (records whose phase was later edited from N/A, Phase 4 or Early Phase 1 into the modelled Phase 1–3; genuine edits) and `primary_purpose` 0.03%.

## Result

| Trials | n | Terminated | Latest-record AUC (95% CI) | Point-in-time AUC (95% CI) | Drop (paired 95% CI) |
|---|---|---|---|---|---|
| with an archived record | 18,568 | 3,600 | 0.704 (0.695–0.712) | 0.699 (0.690–0.707) | 0.005 (0.002 to 0.009) |
| record predates start | 9,112 | 2,192 | 0.655 (0.642–0.666) | 0.649 (0.636–0.660) | 0.006 (0.001 to 0.012) |

Calibration (intercept, slope), latest vs point-in-time: 0.44, 1.05 vs 0.46, 1.02 on all matched trials; 0.53, 0.91 vs 0.57, 0.86 on the strict subset. The large positive intercepts are the censoring of recent starts (only finished trials have labels, and terminated ones finish first, so resolved 2017–2020 trials over-represent terminations: 19.4% here vs 14.8% in the 2015–2016 test set), not the swap. They move little between the two versions.

The strict subset scores lower than all matched trials (0.655 vs 0.704 on latest-record features) because it is a different population: trials registered before they started (with quarterly archives, mostly well before), not a random half of the cohort.

### Which fields carry it

AUC drop when only that column is swapped to its archived value, and how often it changed between the archived record and now:

| Field | Changed, terminated | Changed, completed | Gap (points) | AUC drop, this column alone |
|---|---|---|---|---|
| `criteria_count` | 52.8% | 42.9% | +10.0 | 0.000 |
| `n_countries` | 39.8% | 32.9% | +7.0 | 0.002 |
| `criteria_chars` | 89.1% | 82.7% | +6.4 | 0.000 |
| `us_only` | 16.4% | 11.3% | +5.2 | 0.004 |
| `number_of_arms` | 15.2% | 10.9% | +4.3 | 0.000 |
| `allocation` | 8.9% | 5.3% | +3.6 | 0.000 |
| `n_keywords` | 10.1% | 6.6% | +3.5 | 0.000 |
| `responsible_party` | 5.2% | 3.0% | +2.2 | 0.000 |
| `n_collaborators` | 9.6% | 7.7% | +1.8 | 0.001 |
| `has_dmc` | 5.4% | 3.9% | +1.5 | 0.000 |
| `text` | 100% | 100% | 0 | −0.001 |

Swapping the four fields over the 5-point line together (`criteria_count`, `n_countries`, `criteria_chars`, `us_only`): drop 0.006 (0.004–0.008) on all matched trials, 0.010 (0.005–0.014) on the strict subset. That is all of the overall drop; the other fields' swaps net to about zero.

- Geography carries most of it. Country lists change after start for a third of trials, more often for those that end up terminated, and the model reads the final list.
- The criteria fields have the largest gaps but little AUC weight each, and part of their change rate is residual formatting the unwrap does not catch (it affects both groups alike, so it inflates the rates, not the gap).
- The text changes for every trial (format), and its archived version scores slightly better (−0.001): no sign of text leakage.

## Decision

The plan's rule (Task 7 Step 5):

1. Drop ≤ 0.01 and no field with a terminated-minus-completed change gap above 5 points → keep v3.
2. Specific fields carry the gap → remove or point-in-time-ify those, retrain (v4), re-audit.
3. Broad drop → full point-in-time rebuild before any percentage is shown as a probability.

The drop is under 0.01 on both subsets, but four fields exceed the 5-point gap and together account for the whole drop, so this is **outcome 2**. Archives start in 2017, so those fields cannot be rebuilt as of start for the 2008–2016 training and test years; v4 should drop them. Expect a small cost on the 2015–2016 headline (the stable-only model, which also drops text and four more fields, scores 0.686 vs 0.714) and re-run this audit on v4. Outcome 3 is ruled out: the drop is small and narrow, not broad.

What v3's report can say meanwhile: post-start edits inflate its AUC by about 0.005 on 2017–2020 starts (at most about 0.01 on records registered before start), through four named fields.

## v4 re-audit (2026-10-02)

v4 is v3 without `criteria_count`, `criteria_chars`, `n_countries` and `us_only` (`POST_START_LEAKS` in `src/ctrisk/ml/features.py`), audited on the same reduced archives (`models/v4/audit_point_in_time.json`).

| Trials | n | Latest-record AUC | Point-in-time AUC | Change (paired 95% CI) |
|---|---|---|---|---|
| with an archived record | 18,568 | 0.694 | 0.696 | +0.002 (−0.001 to +0.004) |
| record predates start | 9,112 | 0.645 | 0.648 | +0.004 (−0.001 to +0.008) |

No remaining field moves the AUC by more than 0.0004 when swapped alone. The remaining change gaps (allocation +3.6, responsible party +2.2, collaborators +1.8 points) are under the 5-point line. By the decision rule v4 is the first outcome: keep it, and state the audit in the README.

The cost: v4's 2015–2016 test AUC is 0.703 (0.690–0.716), vs v3's 0.714. That is more than the 0.005 the audit attributed to leakage, because the four fields also carried real signal. Rolling-origin AUCs are 0.719, 0.728, 0.701, 0.703; calibration slope 0.90, intercept 0.13.
