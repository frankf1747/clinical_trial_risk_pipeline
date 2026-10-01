# M6: An Honest Model — Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. **Do not commit** — Frank commits this repo himself. Only Tasks 5 and 7 touch Snowflake; Task 5 also downloads AACT archives.

**Goal:** Make every number and label in the report mean what a reader thinks it means, then measure — before rebuilding anything — how much of the 0.716 AUC depends on registry edits made after trials started.

**Architecture:** Tasks 1–4 change only how results are explained and reported: per-trial drivers carry the feature's value and sign, global importance carries direction, calibration gets a slope and intercept, subgroup AUCs and cohort counts reach the README, and the README stops claiming point-in-time features. Task 5 is the audit: for the 2017–2020 cohort, rebuild the registry-derived features from the AACT monthly archive nearest each trial's start and re-score the saved v2 model on both versions. Task 6 adds the prospective backtest the append-only scores table already makes possible. Task 7 runs it all. The full point-in-time rebuild of the 2008–2016 cohorts is decided by the audit, not assumed.

**Tech Stack:** pandas, scikit-learn, LightGBM `pred_contrib`, PySpark on AACT flat files, Snowflake (DuckDB in tests), pytest.

## Diagnosis this plan acts on

External review of the v2 report (2026-09-29), cross-checked against ClinicalTrials.gov and against this code:

| Finding | Where in the code | Response |
|---|---|---|
| Report says "built only from what was known at start"; page 10 says features describe the latest record. Both cannot be true. | One AACT snapshot (2026-09-26). Latest-record: design fields, `has_dmc`, `start_date` (`clean_trials.py`); countries, eligibility, ages, sex, responsible party, collaborators, keywords (`trial_attributes.py`); the text blob (`trial_text.py`). Point-in-time: only FAERS and sponsor history (`sql/20_features/`). | Task 4 fixes the claim now. Task 5 measures the damage on the cohort AACT archives cover. The rebuild for 2008–2016 follows only if the audit says it must. |
| NCT03801083 and NCT01174121 show "Healthy volunteers" as a main reason; both records say *Accepts Healthy Volunteers: No*. | `score.py::top_drivers` emits bare feature names for positive contributions. `healthy_volunteers` is boolean; the model says *No* raises risk. | Task 1: drivers become `feature=value` with a signed contribution column. |
| "Registration text 0.42, Healthy volunteers 0.23, US-only 0.14" say how much, not which way. | `train.py`: `mean |contribution|`; `registration_text` is the net of 64 SVD columns. | Task 2: signed contribution per level (categoricals, booleans) and per tercile (numerics). |
| Calibration shown only as deciles on the test cohort; no slope, intercept, event counts. | `evaluate.py` has Brier and deciles only. | Task 3: calibration intercept and slope, positives per split, AUC by phase and start year. |
| Missing: sample sizes, termination definition, censoring rule, tuning procedure, text-model construction. | All computed or implied; none surfaced. | Task 4: validation table in the README. |
| "Main reasons" reads as causes; "FDA reports with deaths" reads as a safety measure. | Wording in README and the dashboard (dashboard is not in this repo). | Task 4: "main model contributors"; "historical FAERS reporting signal". |
| No prospective check. | `TRIAL_RISK_SCORES` is append-only with `SCORED_AT`, but scored trials that finish after 2020 starts never get a label. | Task 6: keep labels for every finished trial; `make backtest` ranks scores against later outcomes. |

Facts that shape Task 5: AACT keeps permanent monthly flat-file archives from about 2017; the ClinicalTrials.gov record-history endpoint is undocumented and the `cthist` package that used it reports being broken as of 2026-09-26. So the 2017–2020 cohort (19,596 labeled trials, currently AUC 0.711 on latest-record features) can be audited from stable files; the 2008–2016 cohorts cannot yet.

---

## File map

| File | Responsibility |
|---|---|
| `src/ctrisk/ml/score.py`, `tests/test_ml_score.py` | Drivers with value and contribution; three new columns in `TRIAL_RISK_SCORES` |
| `src/ctrisk/ml/train.py`, `tests/test_ml_train.py` | Driver directions, calibration fit, subgroup AUCs by sponsor class, phase, start year |
| `src/ctrisk/ml/evaluate.py`, `tests/test_ml_evaluate.py` | `calibration_fit`, `positives`, `subgroup_auc` |
| `README.md` | Opening paragraph, validation table, wording |
| `src/ctrisk/spark/clean_trials.py`, `tests/test_clean_trials.py` | `study_fields` (unfiltered per-study columns) reused by the audit; finished trials keep labels past 2020 |
| `src/ctrisk/ingest/aact.py` | Optional destination folder, one per archive |
| `src/ctrisk/spark/point_in_time.py`, `tests/test_point_in_time.py` | Per-archive registry features for a cohort; pick the archive nearest each start |
| `src/ctrisk/ml/audit.py`, `tests/test_ml_audit.py` | Re-score v2 on latest vs point-in-time features; change rates by label |
| `sql/20_features/40_trial_features.sql`, `tests/test_features_sql.py` | `later` split for labeled trials started after 2020 |
| `src/ctrisk/ml/backtest.py` | Scores written earlier vs outcomes known now |
| `Makefile`, `.env.example` | `make audit`, `make backtest`, archive list |

---

### Task 1: Drivers carry their value and sign

**Files:** Modify `src/ctrisk/ml/score.py`, `tests/test_ml_score.py`

- [x] **Step 1: Write failing tests** — replace `tests/test_ml_score.py`:

```python
import numpy as np
import pandas as pd

from ctrisk.ml.score import describe, risk_deciles, top_drivers

NAMES = ["a", "b", "c", "d"]
FRAME = pd.DataFrame({"a": [True, False], "b": [1.0, 2.0], "c": ["PHASE2", "PHASE3"], "d": [None, 3.5]})


def test_top_drivers_are_the_largest_contributions_toward_termination_with_their_values():
    contrib = np.array([[0.5, -0.9, 0.1, 0.3],
                        [-0.2, 0.4, 0.8, 0.0]])
    assert top_drivers(contrib, NAMES, FRAME, k=2) == [[("a=Yes", 0.5), ("d=missing", 0.3)],
                                                       [("c=PHASE3", 0.8), ("b=2", 0.4)]]


def test_drivers_only_list_features_that_raise_risk():
    contrib = np.array([[0.5, -0.9, -0.1, -0.3]])
    assert top_drivers(contrib, NAMES, FRAME.iloc[:1]) == [[("a=Yes", 0.5), None, None]]


def test_describe_shows_booleans_as_yes_no_and_text_as_a_composite():
    assert describe("healthy_volunteers", False) == "healthy_volunteers=No"
    assert describe("n_countries", 12.0) == "n_countries=12"
    assert describe("faers_death_share", 0.0417) == "faers_death_share=0.0417"
    assert describe("registration_text", float("nan")) == "registration_text (net of all text components)"


def test_deciles_run_from_1_lowest_to_10_highest():
    d = risk_deciles(np.linspace(0, 1, 100))
    assert (d.min(), d.max()) == (1, 10)
    assert (d == 10).sum() == 10 and d[-1] == 10
```

- [x] **Step 2: Run to verify failure** — `uv run pytest tests/test_ml_score.py -v` → FAIL (`describe` missing; `top_drivers` signature).

- [x] **Step 3: Implement** — in `src/ctrisk/ml/score.py` replace `top_drivers` with:

```python
import numbers

COMPOSITE = {"registration_text": "registration_text (net of all text components)"}


def describe(feature: str, value) -> str:
    """'healthy_volunteers', False -> 'healthy_volunteers=No'. A reader must never have to guess the value."""
    if feature in COMPOSITE:
        return COMPOSITE[feature]
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        shown = "missing"
    elif isinstance(value, (bool, np.bool_)):
        shown = "Yes" if value else "No"
    elif isinstance(value, numbers.Number):            # int, float, numpy scalars, Decimal from Snowflake
        shown = f"{float(value):g}"
    else:
        shown = str(value)
    return f"{feature}={shown}"


def top_drivers(contrib: np.ndarray, columns: list[str], frame: pd.DataFrame,
                k: int = 3) -> list[list[tuple[str, float] | None]]:
    """Per trial, up to k features pushing its risk up the most, as (feature=value, contribution).
    None where fewer than k push up. Contributions are log-odds from LightGBM's pred_contrib."""
    values = frame.reindex(columns=columns).to_dict("records")     # composites come back as NaN
    out = []
    for row, trial in zip(contrib, values):
        picks = []
        for i in np.argsort(-row)[:k]:
            picks.append((describe(columns[i], trial[columns[i]]), round(float(row[i]), 4)) if row[i] > 0 else None)
        out.append(picks)
    return out
```

In the `__main__` block, replace the driver columns:

```python
        drivers = top_drivers(contrib, names, frame)
        driver_cols = {}
        for n in (1, 2, 3):
            driver_cols[f"TOP_DRIVER_{n}"] = [d[n - 1][0] if d[n - 1] else None for d in drivers]
            driver_cols[f"TOP_DRIVER_{n}_CONTRIB"] = [d[n - 1][1] if d[n - 1] else None for d in drivers]
        scores = pd.DataFrame({
            "NCT_ID": frame["nct_id"], "MODEL_VERSION": f"v{version}",
            "RISK_SCORE": p.round(4), "RISK_DECILE": risk_deciles(p),
            "ENROLLMENT_RISK_SCORE": enrollment.predict_proba(frame).round(4),
            **driver_cols,
            "SCORED_AT": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")})
        for col in ("ENROLLMENT_RISK_SCORE", "TOP_DRIVER_1_CONTRIB", "TOP_DRIVER_2_CONTRIB", "TOP_DRIVER_3_CONTRIB"):
            conn.cursor().execute(f"ALTER TABLE IF EXISTS TRIAL_RISK_SCORES ADD COLUMN IF NOT EXISTS {col} FLOAT")
```

Docstring for the module: `"""Score active trials with the latest version: overall and enrollment termination risk, plus the top model contributors (feature=value, signed contribution)."""` Earlier score rows keep bare names in `TOP_DRIVER_n`; the `MODEL_VERSION` and `SCORED_AT` columns tell them apart.

- [x] **Step 4: Run** — `uv run pytest tests/test_ml_score.py -v && uv run ruff check src tests` → pass.

---

### Task 2: Global importance with direction

**Files:** Modify `src/ctrisk/ml/train.py`, `tests/test_ml_train.py`

- [x] **Step 1: Write failing tests** — append to `tests/test_ml_train.py`:

```python
def test_top_drivers_say_which_way_each_feature_pushes(result):
    _, report = result
    top = {d["feature"]: d for d in report["top_drivers"]}
    n = top["n_countries"]["direction"]                       # the synthetic signal: more -> terminated
    assert n["high_third"] > 0 > n["low_third"]
    assert set(top["phase"]["direction"]) <= {"PHASE1", "PHASE2", "PHASE3", "missing"}
    assert top["registration_text"]["direction"] is None      # a composite has no single value
```

- [x] **Step 2: Run to verify failure** — `uv run pytest tests/test_ml_train.py -v -k direction` → KeyError.

- [x] **Step 3: Implement** — in `src/ctrisk/ml/train.py` add:

```python
def driver_directions(contrib: np.ndarray, names: list[str], frame: pd.DataFrame) -> dict[str, dict | None]:
    """Mean signed contribution by feature value: per level for categoricals and booleans, for the lowest
    and highest third for numerics. Positive pushes toward termination. None for composites (text)."""
    out = {}
    for j, name in enumerate(names):
        if name not in frame.columns:
            out[name] = None
            continue
        c, v = pd.Series(contrib[:, j], index=frame.index), frame[name]
        if name in CATEGORICAL or v.dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all():
            levels = v.astype(object).where(v.notna(), "missing").astype(str)
            out[name] = {lvl: round(float(c[levels == lvl].mean()), 4) for lvl in sorted(levels.unique())}
        else:
            x = v.map(lambda a: np.nan if pd.isna(a) else float(a))
            lo, hi = x.quantile([1 / 3, 2 / 3])
            out[name] = {"low_third": round(float(c[x <= lo].mean()), 4),
                         "high_third": round(float(c[x >= hi].mean()), 4)}
    return out
```

and in `run()` replace the `top_drivers` block:

```python
    contrib, names = collapse_text(main.contributions(test), main.feature_names)
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=names)     # size only; direction below
    directions = driver_directions(contrib, names, test)
    # registration_text is the net of 64 SVD columns, so its size is not comparable one-to-one with a single feature
    report["top_drivers"] = [{"feature": f, "mean_abs_contribution": round(float(v), 4), "direction": directions[f]}
                             for f, v in importance.sort_values(ascending=False).head(15).items()]
```

- [x] **Step 4: Run** — `uv run pytest tests/test_ml_train.py -v && uv run ruff check src tests` → pass.

---

### Task 3: Calibration fit, event counts, subgroup AUCs

**Files:** Modify `src/ctrisk/ml/evaluate.py`, `tests/test_ml_evaluate.py`, `src/ctrisk/ml/train.py`, `tests/test_ml_train.py`

- [x] **Step 1: Write failing tests** — append to `tests/test_ml_evaluate.py`:

```python
def test_metrics_count_positives():
    assert metrics(Y, PERFECT)["positives"] == 2


def test_calibration_fit_is_identity_for_a_calibrated_score():
    from ctrisk.ml.evaluate import calibration_fit
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, 20000)
    y = (rng.random(20000) < p).astype(int)
    fit = calibration_fit(y, p)
    assert abs(fit["slope"] - 1) < 0.05 and abs(fit["intercept"]) < 0.05


def test_calibration_fit_flags_an_overconfident_score():
    from ctrisk.ml.evaluate import calibration_fit
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, 20000)
    y = (rng.random(20000) < p).astype(int)
    logit = np.log(p / (1 - p))
    too_sure = 1 / (1 + np.exp(-2 * logit))          # same ranking, twice the confidence
    assert 0.4 < calibration_fit(y, too_sure)["slope"] < 0.6


def test_subgroup_auc_skips_small_or_single_class_groups():
    import pandas as pd
    from ctrisk.ml.evaluate import subgroup_auc
    y = pd.Series([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    p = pd.Series(PERFECT)
    groups = pd.Series(["big"] * 8 + ["small"] * 2)
    out = subgroup_auc(y, p, groups, min_n=5)
    assert set(out) == {"big"} and out["big"] == {"n": 8, "positives": 2, "roc_auc": 1.0}
```

- [x] **Step 2: Run to verify failure** — `uv run pytest tests/test_ml_evaluate.py -v` → FAIL.

- [x] **Step 3: Implement** — in `src/ctrisk/ml/evaluate.py`:

```python
from scipy.optimize import brentq                    # scipy comes with scikit-learn
from sklearn.linear_model import LogisticRegression
```

add `"positives": int(y.sum()),` to `metrics()` after `"n"`, and append:

```python
def calibration_fit(y, score) -> dict:
    """Calibration-in-the-large and slope (Steyerberg). Intercept 0 and slope 1 mean the probabilities
    can be read as they are; intercept > 0 means risk is under-predicted overall; slope < 1 means
    the score is more extreme than the outcomes justify."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    slope = LogisticRegression(penalty=None, max_iter=1000).fit(logit[:, None], y).coef_[0][0]
    intercept = brentq(lambda a: (1 / (1 + np.exp(-(logit + a)))).mean() - y.mean(), -10, 10)  # slope fixed at 1
    return {"intercept": round(float(intercept), 4), "slope": round(float(slope), 4)}


def subgroup_auc(y: pd.Series, score: pd.Series, groups: pd.Series, min_n: int = 100) -> dict:
    """ROC AUC per group, for groups big enough to mean something and with both outcomes present."""
    out = {}
    for key, idx in groups.groupby(groups, dropna=False).groups.items():
        if len(idx) >= min_n and y[idx].nunique() > 1:
            out["missing" if pd.isna(key) else str(key)] = {
                "n": int(len(idx)), "positives": int(y[idx].sum()),
                "roc_auc": round(float(roc_auc_score(y[idx], score[idx])), 4)}
    return out
```

In `src/ctrisk/ml/train.py`: import `calibration_fit, subgroup_auc`; in `evaluate()` under `if full:` add `out["calibration_fit"] = calibration_fit(y, p)`; replace the `by_sponsor_class` block with:

```python
    p = pd.Series(models["lightgbm"].predict_proba(test), index=test.index)
    yt = y[rows["test"]]
    report["by_sponsor_class"] = subgroup_auc(yt, p, test["sponsor_class"]) if "sponsor_class" in columns else {}
    report["by_phase"] = subgroup_auc(yt, p, test["phase"]) if "phase" in columns else {}
    report["by_start_year"] = subgroup_auc(yt, p, pd.to_datetime(test["start_date"]).dt.year)
```

and print `calibration_fit`, `by_phase`, `by_start_year` in the `__main__` summary next to the existing lines. In `tests/test_ml_train.py::test_ablations_and_subgroups` add `assert set(label["by_phase"]) == {"PHASE1", "PHASE2", "PHASE3"}` and `assert "calibration_fit" in label["models"]["lightgbm"]["test"]`. (The synthetic test split has ~1,000 rows, so each phase clears `min_n=100`; start years have ~140 rows each.)

- [x] **Step 4: Run** — `uv run pytest -q && uv run ruff check src tests` → pass.

---

### Task 4: The README says what the model is

**Files:** Modify `README.md`

- [x] **Step 1: Opening** — replace the first two lines with:

```
# Clinical Trial Risk Pipeline

Ranks Phase 1–3 drug trials by their risk of early termination, from the registry record, the sponsor's track record before the trial started, and the drug's FAERS reporting history before the trial started.

Caveat first: sponsor and FAERS features are computed as of each trial's start date, but the registry-record features (design, eligibility, geography, text) come from the latest version of the record, because the AACT snapshot holds only that version. Records are edited during trials, so the held-out AUC below may be inflated by post-start edits. M6 measures this (`docs/plans/2026-09-29-m6-honest-model.md`); until then treat the numbers as preliminary and the per-trial percentages as rankings, not probabilities.
```

- [x] **Step 2: Validation summary** — after the `## Results` table add:

```
### Validation summary (v2)

| | Train | Test | Recent |
|---|---|---|---|
| Start years | 2008–2014 | 2015–2016 | 2017–2020 |
| Trials | 36,286 | 10,242 | 19,596 |
| Terminated | 5,055 (13.9%) | 1,513 (14.8%) | 3,659 (18.7%) |
| Use | fit; tuned on <2013 vs 2013–2014 | headline metrics, calibration | reported with censoring caveat |

- **Outcome:** `overall_status = TERMINATED` (1) vs `COMPLETED` (0). Withdrawn, suspended, unknown-status and still-running trials carry no label and are excluded from every split; the recent years therefore over-represent early terminations (terminated trials finish sooner).
- **Never touched by tuning or text fitting:** the 2015–2016 rows. Tuning uses an inner time split inside the training years; TF-IDF/SVD is fit on training rows only (`tests/test_ml_train.py` pins both).
- **Text model:** TF-IDF over word 1–2grams (min_df 20, ≤50k terms), TruncatedSVD to 64 components, fit on training rows only.
- **Missing data:** LightGBM's native handling; median imputation plus missing indicators for the logistic baseline.
- **Calibration (test):** Brier 0.117; deciles in `models/v2/metrics.json`; intercept and slope from the next `make train`.
- **By subgroup (test AUC):** industry 0.743, academic/other 0.672, government 0.671; by phase and start year from the next `make train`.
```

(Counts are `n × base_rate` from `models/v2/metrics.json`, rounded; replace with the exact `positives` after Task 7.)

- [x] **Step 3: Wording** — in `## Results` and `## Status`: "top drivers" → "main model contributors (SHAP contributions, not causes)"; FAERS features described as "historical FAERS reporting signal: FAERS has duplicate and incomplete reports and cannot establish causation or incidence (FDA)"; the enrollment-driven precision line keeps its numbers. Keep the existing Limitations paragraph, minus the latest-record sentence now in the opening.

- [x] **Step 4: Check** — the README no longer contains the phrase "using only what is known when they start".

---

### Task 5: Point-in-time audit on the 2017–2020 cohort

Cost: 17 quarterly AACT archives (2017-01 … 2021-01, ~1.5 GB each zipped; delete each zip after extracting the 12 tables), ~1 h of local Spark, one Snowflake read. No GCS.

**Files:** Modify `src/ctrisk/ingest/aact.py`, `src/ctrisk/spark/clean_trials.py`, `tests/test_clean_trials.py`, `Makefile`, `.env.example`; create `src/ctrisk/spark/point_in_time.py`, `tests/test_point_in_time.py`, `src/ctrisk/ml/audit.py`, `tests/test_ml_audit.py`

- [x] **Step 1: One folder per archive** — `src/ctrisk/ingest/aact.py::__main__` accepts an optional second argument:

```python
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python -m ctrisk.ingest.aact <url-or-zip-path> [dest-folder]")
    cfg = load_config()
    dest = Path(sys.argv[2]) if len(sys.argv) == 3 else Path(cfg.local_root) / "raw" / "aact"
    for p in fetch(sys.argv[1], dest):
```

Makefile: `ingest-aact` passes `$(DEST)` through. `.env.example` gains a comment listing the archive URL pattern to copy from the AACT snapshots page (`https://aact.ctti-clinicaltrials.org/downloads/snapshots?type=flatfiles&year=YYYY`; file names carry the date).

- [x] **Step 2: Reusable per-study fields** — in `src/ctrisk/spark/clean_trials.py` split `build_trials`:

```python
def study_fields(studies: DataFrame, designs: DataFrame, sponsors: DataFrame) -> DataFrame:
    """Every study's registry-record fields the model uses, unfiltered: the audit reads these from
    archived snapshots; build_trials filters them to the eligible population."""
    (the current `lead_sponsor`, `design` and `s` selects, joined on nct_id with left joins,
     plus `F.col("start_date_type")` from studies, kept for the audit)


def build_trials(...):
    s = study_fields(studies, designs, sponsors)
    (the current eligibility filter, drug join, label, stop_reason, drop of study_type, why_stopped, start_date_type)
```

Test: `tests/test_clean_trials.py` gains `test_study_fields_keeps_every_study(aact)` asserting all ten fixture studies come back with `status`, `phase`, `sponsor_class`. Add `start_date_type` to the studies fixture header (values `ACTUAL` for finished rows, `ANTICIPATED` for NCT003, blank elsewhere); the existing tests do not read it.

- [x] **Step 3: Write failing tests** — `tests/test_point_in_time.py`, on the fixture treated as two archives:

```python
import pandas as pd

from ctrisk.spark.point_in_time import nearest_archive, registry_features


def test_registry_features_cover_only_the_cohort(aact):
    cohort = aact["studies"].select("nct_id").where("nct_id IN ('NCT001', 'NCT002', 'NCT404')")
    rows = registry_features(aact, cohort, archive_date="2013-01-01").collect()
    assert {r.nct_id for r in rows} == {"NCT001", "NCT002"}         # NCT404 is not in this archive
    r = {x.nct_id: x for x in rows}["NCT001"]
    assert (r.archive_date, r.phase, r.healthy_volunteers, r.n_countries) == ("2013-01-01", "PHASE2", False, 1)
    assert "Pembrolizumab" in r.text


def test_nearest_archive_is_the_first_on_or_after_start_else_the_earliest_seen():
    seen = pd.DataFrame({"nct_id": ["A", "A", "B", "B"],
                         "archive_date": pd.to_datetime(["2017-01-01", "2017-04-01", "2017-04-01", "2017-07-01"]),
                         "x": [1, 2, 3, 4]})
    starts = pd.Series(pd.to_datetime(["2017-02-10", "2017-01-01"]), index=["A", "B"])
    pick = nearest_archive(seen, starts).set_index("nct_id")
    assert pick.loc["A", "x"] == 2 and pick.loc["A", "lag_days"] == 50
    assert pick.loc["B", "x"] == 3 and pick.loc["B", "lag_days"] == 90   # registered after start
```

- [x] **Step 4: Implement** — `src/ctrisk/spark/point_in_time.py`:

```python
"""Registry-record features for a cohort as they stood in archived AACT snapshots.

For each archive: the same design, attribute and text fields the pipeline builds from the current
snapshot, for the cohort's trials present in that archive, tagged with the archive date. Then, per
trial, the first archive dated on or after its start (the record as the trial began), or the
earliest archive that has it at all when it was registered late. lag_days says how far off that is.
"""
import sys
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ctrisk.config import load_config
from ctrisk.spark.clean_trials import read_table, study_fields
from ctrisk.spark.session import get_spark
from ctrisk.spark.trial_attributes import build_trial_attributes
from ctrisk.spark.trial_text import build_trial_text

TABLES = ("studies", "designs", "sponsors", "countries", "eligibilities", "browse_conditions",
          "responsible_parties", "keywords", "brief_summaries")


def registry_features(t: dict, cohort: DataFrame, archive_date: str) -> DataFrame:
    ids = cohort.select("nct_id").distinct()
    fields = study_fields(t["studies"], t["designs"], t["sponsors"]).join(ids, "nct_id")
    attrs = build_trial_attributes(fields, t["countries"], t["eligibilities"], t["browse_conditions"],
                                   t["sponsors"], t["responsible_parties"], t["keywords"])
    text = build_trial_text(fields, t["studies"], t["brief_summaries"], t["eligibilities"], t["keywords"])
    return (fields.join(attrs, "nct_id", "left").join(text, "nct_id", "left")
            .withColumn("archive_date", F.lit(archive_date)))


def nearest_archive(seen: pd.DataFrame, starts: pd.Series) -> pd.DataFrame:
    """One row per trial: its record from the first archive on/after start, else the earliest archive."""
    seen = seen.assign(start=seen["nct_id"].map(starts))
    seen["lag_days"] = (seen["archive_date"] - seen["start"]).dt.days
    on_or_after = seen[seen["lag_days"] >= 0].sort_values("archive_date").groupby("nct_id").head(1)
    rest = seen[~seen["nct_id"].isin(on_or_after["nct_id"])].sort_values("archive_date").groupby("nct_id").head(1)
    return pd.concat([on_or_after, rest]).drop(columns="start").reset_index(drop=True)


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("point_in_time", cfg.mode)
    trials = spark.read.parquet(cfg.path("parquet", "trials"))
    cohort = trials.where("label IS NOT NULL AND start_date >= DATE '2017-01-01' AND start_date < DATE '2021-01-01'")
    archives = sorted(Path(cfg.local_root, "raw", "aact_archive").glob("*"))    # folders named YYYY-MM-DD
    for folder in archives:
        t = {name: read_table(spark, str(folder), name) for name in TABLES}
        registry_features(t, cohort, folder.name).write.mode("overwrite") \
            .parquet(cfg.path("parquet_pit", "by_archive", folder.name))
    seen = spark.read.option("basePath", cfg.path("parquet_pit", "by_archive")) \
        .parquet(cfg.path("parquet_pit", "by_archive", "*")).toPandas()
    seen["archive_date"] = pd.to_datetime(seen["archive_date"])
    starts = trials.select("nct_id", "start_date").toPandas().set_index("nct_id")["start_date"]
    picked = nearest_archive(seen, pd.to_datetime(starts))
    picked.to_parquet(cfg.path("parquet_pit", "registry_at_start.parquet"), index=False)
    print({"cohort": cohort.count(), "with_record": len(picked),
           "lag_days_median": float(picked["lag_days"].median()),
           "registered_after_start": int((picked["lag_days"] < 0).sum())})
```

Makefile: `audit-build: uv run python -m ctrisk.spark.point_in_time`.

- [x] **Step 5: The comparison** — `tests/test_ml_audit.py` first:

```python
import numpy as np
import pandas as pd

from ctrisk.ml.audit import change_rates, overlay


def test_overlay_replaces_registry_columns_and_keeps_point_in_time_ones():
    latest = pd.DataFrame({"nct_id": ["A", "B"], "phase": ["PHASE2", "PHASE3"], "faers_reports": [5, 0],
                           "text": ["new", "new"], "label": [1, 0]})
    pit = pd.DataFrame({"nct_id": ["A"], "phase": ["PHASE1"], "text": ["old"]})
    out = overlay(latest, pit, registry=["phase", "text"])
    assert out.loc[out.nct_id == "A", "phase"].item() == "PHASE1" and out["faers_reports"].tolist() == [5, 0]
    assert len(out) == 1                                   # trials without an archived record are dropped


def test_change_rates_by_label():
    latest = pd.DataFrame({"nct_id": list("ABCD"), "x": [1, 1, 1, 1], "label": [1, 1, 0, 0]})
    pit = pd.DataFrame({"nct_id": list("ABCD"), "x": [0, 1, 1, 1]})
    assert change_rates(latest, pit, ["x"]) == {"x": {"terminated": 0.5, "completed": 0.0}}
```

then `src/ctrisk/ml/audit.py`:

```python
"""Does the held-out AUC survive when registry features come from the record at trial start?

Scores the saved model twice on the same trials: once on TRIAL_FEATURES (latest record), once with the
registry-derived columns replaced by their values in the AACT archive nearest each start date.
FAERS and sponsor-history columns are already point-in-time and stay as they are.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ctrisk.ml.evaluate import bootstrap_auc_ci, calibration, calibration_fit, metrics

REGISTRY = ["phase", "number_of_arms", "allocation", "intervention_model", "primary_purpose", "masking",
            "sponsor_class", "has_dmc", "n_countries", "us_only", "min_age_years", "max_age_years",
            "healthy_volunteers", "sex", "criteria_count", "criteria_chars", "responsible_party",
            "n_collaborators", "n_keywords", "text"]          # plus every area_* column, added at runtime


def overlay(latest: pd.DataFrame, pit: pd.DataFrame, registry: list[str]) -> pd.DataFrame:
    cols = [c for c in registry if c in pit.columns]
    out = latest.drop(columns=cols).merge(pit[["nct_id", *cols]], on="nct_id", how="inner")
    return out[latest.columns]


def change_rates(latest: pd.DataFrame, pit: pd.DataFrame, columns: list[str]) -> dict:
    both = latest.merge(pit, on="nct_id", suffixes=("", "_pit"))
    out = {}
    for c in columns:
        differs = ~((both[c] == both[f"{c}_pit"]) | (both[c].isna() & both[f"{c}_pit"].isna()))
        out[c] = {"terminated": round(float(differs[both["label"] == 1].mean()), 4),
                  "completed": round(float(differs[both["label"] == 0].mean()), 4)}
    return out


def compare(model, latest: pd.DataFrame, pit: pd.DataFrame, registry: list[str]) -> dict:
    y = latest.set_index("nct_id")["label"]
    a = model.predict_proba(latest)
    swapped = overlay(latest, pit, registry)
    b = model.predict_proba(swapped)
    ya, yb = y[latest["nct_id"]].to_numpy(), y[swapped["nct_id"]].to_numpy()
    return {"n_latest": len(latest), "n_point_in_time": len(swapped),
            "latest": {**metrics(ya, a), "roc_auc_ci95": bootstrap_auc_ci(ya, a), "calibration_fit": calibration_fit(ya, a)},
            "point_in_time": {**metrics(yb, b), "roc_auc_ci95": bootstrap_auc_ci(yb, b),
                              "calibration_fit": calibration_fit(yb, b), "calibration": calibration(yb, b)},
            "change_rates": change_rates(latest, pit, [c for c in registry if c in pit.columns])}
```

`__main__`: load the latest model version (`registry.latest/load`), read `SELECT * FROM TRIAL_FEATURES_V{version} WHERE split = 'recent'` from Snowflake (columns lower-cased, `start_date` to datetime), read `data/parquet_pit/registry_at_start.parquet`, set `registry = REGISTRY + [c for c in latest.columns if c.startswith("area_")]`, run `compare`, write `models/v{version}/audit_point_in_time.json` and print the two AUCs, the CI, the calibration fits, and the ten features whose terminated-vs-completed change rates differ most. Makefile: `audit: uv run python -m ctrisk.ml.audit`.

- [x] **Step 6: Run** — `uv run pytest -q && uv run ruff check src tests` → pass.

---

### Task 6: Labels for every finished trial, and a prospective backtest

**Files:** Modify `src/ctrisk/spark/clean_trials.py`, `tests/test_clean_trials.py`, `sql/20_features/40_trial_features.sql`, `tests/test_features_sql.py`, `Makefile`; create `src/ctrisk/ml/backtest.py`

- [x] **Step 1: Keep later finishers** — in `clean_trials.py` drop the upper bound: `finished = status in (COMPLETED, TERMINATED) & (start_date >= TRAIN_START)`; delete `TRAIN_END`. `test_keeps_only_eligible_drug_trials` now expects `NCT010` too (Phase 2, completed, 2021 start; check its intervention row is a drug — if not, add one to `interventions.txt`). `test_label_is_terminated_vs_completed_and_null_for_active` gains `"NCT010": 0`.
- [x] **Step 2: A `later` split** — `40_trial_features.sql`: `WHEN t.start_date < DATE '2021-01-01' THEN 'recent' ELSE 'later' END`. `train.py` already reads only `train`, `test`, `recent`; its label check (`split != 'score'` rows must have labels) still holds. `tests/test_features_sql.py`: add `('T9', DATE '2021-02-01', 0, NULL, 'Eps', 'OTHER')` and assert `features(db, "T9")["split"] == "later"`. Snowflake's `90_checks.sql` needs no change.
- [x] **Step 3: Backtest** — `src/ctrisk/ml/backtest.py`:

```python
"""Prospective check: how well did scores written earlier rank the trials that have since finished?

Every `make score` appends to TRIAL_RISK_SCORES; every `make warehouse` on a fresh AACT snapshot
relabels trials that finished since. Joining the two is a true forward test, with one caveat: the
trials resolved so far skew toward early terminations, so precision is inflated and AUC slightly so.
Re-run yearly; the caveat fades as the cohort resolves.
"""
QUERY = """
SELECT s.nct_id, s.model_version, s.scored_at, s.risk_score, t.label
FROM TRIAL_RISK_SCORES s JOIN TRIAL_FEATURES t ON t.nct_id = s.nct_id
WHERE t.label IS NOT NULL"""
```

`__main__`: read the query, group by `model_version`, print `n`, `positives`, `roc_auc`, `roc_auc_ci95`, `precision_top_10pct`, `calibration_fit`, plus the count of scored trials still unresolved (`SELECT COUNT(DISTINCT nct_id) FROM TRIAL_RISK_SCORES WHERE nct_id NOT IN (SELECT nct_id FROM TRIAL_FEATURES WHERE label IS NOT NULL)`). Write `models/backtest_{YYYY-MM-DD}.json`. Makefile: `backtest: uv run python -m ctrisk.ml.backtest`. Pure logic is the existing `evaluate.py`; no new unit test beyond the SQL one in Step 2.

- [x] **Step 4: Run** — `uv run pytest -q && uv run ruff check src tests` → pass.

---

### Task 7: Real run (coordinator)

- [x] **Step 1: Retrain for the reporting changes** — `make train` (v3; same data as v2, so expect AUC 0.716 ± noise from the identical seed; if it moves by more than 0.005 something else changed). Record `calibration_fit`, `by_phase`, `by_start_year`, driver directions. Sanity: `healthy_volunteers` direction should show `False` positive and `True` negative; `us_only` should now say which way. Also record the stable-only AUC (`lightgbm_stable_only`) and the four rolling-origin windows, then `make report` for `docs/model_card.md`. Read the stable-only AUC before the audit: if it is within ~0.02 of the full model, post-start edits can explain at most that much of the headline, whatever the audit finds; if it drops far more, the audit decides which of those fields are leaking and which are simply informative.
- [x] **Step 2: Score** — `make score`; in Snowflake: `SELECT TOP_DRIVER_1, ROUND(AVG(TOP_DRIVER_1_CONTRIB),3), COUNT(*) FROM TRIAL_RISK_SCORES WHERE MODEL_VERSION='v3' GROUP BY 1 ORDER BY 3 DESC LIMIT 15`. Check NCT03801083 and NCT01174121 read `healthy_volunteers=No`.
- [ ] **Step 3: Archives** (partial: 2017's 12 monthly archives done, 2018-01 … 2021-01 to go; see run log below) — download the 17 quarterly flat-file archives 2017-01-01 … 2021-01-01 into `data/raw/aact_archive/<YYYY-MM-DD>/` with `make ingest-aact AACT=<zip> DEST=data/raw/aact_archive/<date>`; delete each zip after extraction. Note in the plan which months were actually available (the archive page decides).
- [ ] **Step 4: Audit** (run on the 2017 archives only: 5,482 of 19,596 trials) — `make audit-build` (expect `with_record` ≈ 19,000 of 19,596; median lag ≤ 45 days; a few hundred registered after start) then `make audit`.
- [ ] **Step 5: Decide** (preliminary note `docs/audits/2026-09-30-point-in-time.md`; no decision until all archives are in) — write `docs/audits/2026-MM-DD-point-in-time.md` with: both AUCs and CIs, calibration fits, the change-rate table sorted by the terminated-minus-completed gap, and one of three decisions:
  - drop ≤ 0.01 and no feature with a label-dependent change gap above 5 points → keep v3, keep the caveat, state the audit result in the README;
  - specific features carry the gap (expect `criteria_count`, `criteria_chars`, `text`, `n_countries`) → remove or point-in-time-ify those, retrain (v4), re-audit;
  - broad drop → the full rebuild (below) is required before any percentage is shown as a probability.
- [x] **Step 5b: Regenerate the report** — `make report` again so `docs/model_card.md` carries the audit tables; `make dashboard` so the page shows values and contributions.
- [x] **Step 6: README** — replace the "from the next `make train`" placeholders with v3 numbers; add the audit paragraph; note that `TOP_DRIVER_n` now reads `feature=value`. Tell Frank what the report's page 1 can truthfully say.
- [x] **Step 7: Backtest baseline** — `make backtest` now, expecting almost no resolved rows (scores are days old); commit the JSON so the first real backtest in 2027 has a starting point.

## After M6

**Full point-in-time rebuild (2008–2016 starts), only if Task 7 Step 5 lands on the third outcome.** Needs ClinicalTrials.gov record history for ~46,000 trials. Verify the current history endpoint in a browser first (the `cthist` package broke on 2026-09-26); design a polite fetcher (one request per second, resumable, cached per trial), take the last version dated on or before each start date, map its fields onto `study_fields`, `trial_attributes` and `trial_text`, retrain, and re-test on 2015–2016. Then the harder schedule the reviewer asked for — train 2008–2018, validate 2019–2021, untouched test 2022–2023 — is possible because Task 6 now labels those years, with the censoring caveat stated per year.

**Not in M6:** full FAERS (S1); planned enrollment from archives (the old M6, now folded into the archive work in Task 5: once archives are ingested, first-registered enrollment is one more column).

## Changes during implementation (Tasks 1–6 done; Task 7 not yet run)

- **Archive choice:** the builder takes each trial's *last* archived record on or before its start date, which cannot contain post-start edits, and falls back to the first record after start only for trials registered late (`lag_days > 0`). The draft above picked the first archive after start, which can carry weeks of edits.
- **Monthly, one archive at a time:** `make audit-build` extracts each archive, keeps the cohort's rows (`data/parquet_pit/by_archive/<date>/`), deletes the extract, and skips archives already reduced, so monthly archives (2017-01 … 2021-01) fit on a laptop. Each download now gets its own zip name; before, `download()` would have reused the first archive's zip for every later one.
- **Legacy registry wording:** archives before the 2023 registry modernization spell enums as display text ("Parallel Assignment", "None (Open Label)", "Sponsor-Investigator", "Accepts Healthy Volunteers"). `design_value`, `masking_value` and `flag` in `clean_trials.py` map both eras to the same values; current data is unchanged. The builder prints per-archive null shares so remaining drift is visible, and the audit reports `unseen_levels`: archived categories the model never trained on.
- **Audit on matched trials:** both AUCs are computed on the same trials (those with an archived record), with a paired bootstrap interval on the drop, plus a strict subset whose record predates the start, and a one-column-at-a-time swap that isolates which column carries the gap. Change rates report the terminated-minus-completed `gap`: leakage shows as a positive gap, format drift as change in both groups.
- **Population:** finished trials now keep labels for starts after 2020 (split `later`, used only by the backtest). The drug match-rate gate counts only modelled trials (`start_date < MODEL_END`, 2021-01-01) so its threshold keeps its meaning; `check_trials` still gates on all labeled trials.
- **Backtest** uses each trial's first score per model version.
- **Dashboard (merged from main's V2 update):** the "Main reasons" column is now "Main model contributors" and renders `feature=value` with the signed contribution ("Healthy volunteers: No (+0.21)"); the global chart's tooltip gives direction per value; the calibration caption shows slope and intercept when present; the lede and notes carry the latest-record caveat instead of "built only from what was known when each trial started"; FAERS labels read as a reporting signal. `30_serving.sql` adds the contribution columns if missing, so pre-M6 scores still read. `docs/dashboard/index.html` was re-rendered from its embedded v2 data, so its per-trial contributors stay bare names until `make score && make dashboard`.
- **Analysis additions (no new data needed; computed on the next `make train`):** `lightgbm_stable_only` drops every edit-prone field (`EDIT_PRONE` in `ml/features.py`: criteria count and length, countries, US-only, collaborators, keywords, responsible party, DMC) and the text, bounding how much of the AUC post-start edits could explain before any archive is read. `rolling_origin` evaluates the tuned model on 2012–13, 2013–14, 2014–15 and 2015–16 starts, each trained only on earlier training-split starts, with calibration per window. `make report` writes `docs/model_card.md`, a TRIPOD+AI-style report generated from the version's metrics, manifest, features, audit and backtest files; items not yet computed read as pending. The dashboard shows the stable-only bar and the rolling windows when present and links the card.

## Task 7 run log (2026-09-30)

- **Tests:** one failure on arrival: `calibration_fit`'s Newton step overshot on a small synthetic fold (232 trials, 6 terminated) into a singular Hessian. Fixed with step-halving on the log-likelihood and a singular-Hessian guard; the slope on that fold matches unpenalized sklearn (0.588).
- **`make m6`:** stopped once at `upload` (expired gcloud login), then ran to the end. Splits unchanged from v2 (train 36,286, test 10,242, recent 19,596), plus 15,100 labeled `later` trials.
- **v3:** test AUC 0.714 (0.700–0.726) vs v2 0.716; early stopping chose 123 trees (v2: 146). Calibration intercept 0.10, slope 1.06. `healthy_volunteers` No +0.15 / Yes −0.42; `us_only` Yes +0.16 / No −0.12. Stable-only 0.686 (−0.028, just over the ~0.02 bar, so the audit decides). Rolling origin 0.718 / 0.734 / 0.712 / 0.714. NCT03801083 and NCT01174121 read `healthy_volunteers=No` (+0.18, +0.15) in `TRIAL_RISK_SCORES`.
- **Archives:** AACT now serves snapshots only after sign-in, as monthly archives under Snapshot history (one per month, 2017 onward; listing dates are not the 1st). Each download redirects to a signed URL, so `AACT_ARCHIVES` takes local zips; each zip was named by the latest `last_changed_date` in its `studies.txt` (`nlm_download_date_description` is per study and misdates archives). 2017's 12 archives are in `data/raw/aact_archive_zips/`; 2018–2021 were blocked by Chrome's multiple-download setting.
- **Format drift fixed (with tests):** start date from `start_month_year`; legacy "Double Blind" levelled by masked-party count; Feb–Aug 2017 party-list masking; blank allocation for single-arm trials → NA; "U.S. Fed" → GOVERNMENT; "Both" → ALL; "Educational/Counseling/Training" → ECT; hard-wrapped criteria unwrapped (`point_in_time.unwrap_criteria`); disease areas unknown where an archive lists no MeSH ancestors (the audit keeps the latest value and reports the share). Rebuilding the current snapshot with these changes alters 0 trial and 0 attribute rows.
- **Audit (partial, 2017 archives):** drop 0.008 (paired 95% CI 0.002–0.015) on 5,482 matched trials; 0.007 (−0.002 to 0.016) on the 3,788 whose record predates start. Carried by `us_only` (0.007) and `n_countries` (0.002). `criteria_count` has a +8.1-point change gap with no AUC effect.
- **Backtest baseline:** `models/backtest_2026-10-01.json`, nothing resolved yet.
