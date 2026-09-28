# M4: Train, Evaluate, and Score — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do not commit** — Frank commits this repo himself. Only Task 8 touches Snowflake.

**Goal:** A versioned LightGBM model with an honest held-out AUC (and confidence interval), explained drivers, and risk scores for every active trial written to Snowflake.

**Architecture:** `train.py` clones `TRIAL_FEATURES` to `TRIAL_FEATURES_V{N}`, reads the clone, fits a logistic-regression baseline and LightGBM (plus two ablations), evaluates on the resolved 2015–2016 test years and on the censored 2017–2020 "recent" years, and saves `models/v{N}/`. `score.py` loads the latest version, scores active trials, and appends to `TRIAL_RISK_SCORES` with each trial's top three risk drivers. All logic is in pure functions tested on synthetic data; Snowflake I/O is a thin shell.

**Tech Stack:** pandas, scikit-learn, LightGBM (drivers via its built-in `pred_contrib`), snowflake-connector-python `[pandas]`, pytest.

**Why this split:** among trials starting 2008–2016, ≤5% are still running and termination rates sit at 13–15%. For 2017–2020 up to 18% are still running and the finished ones skew toward early terminations (17–21%). The headline number comes from resolved years; the recent years are reported separately with that caveat.

**Expectation to set now:** published design-only models land around 0.70–0.80 AUC. The README reports whatever we get; the resume bullet follows the README.

---

## File map

| File | Responsibility |
|---|---|
| `sql/20_features/40_trial_features.sql`, `sql/90_checks.sql`, `tests/test_features_sql.py` | Four-way split: `train` <2015, `test` 2015–2016, `recent` 2017–2020, `score` = active |
| `src/ctrisk/ml/__init__.py` | Empty |
| `src/ctrisk/ml/features.py` | Which columns are inputs; frame → model matrix with fixed categories |
| `src/ctrisk/ml/evaluate.py` | ROC/PR AUC, precision in top 10%, Brier, calibration table, bootstrap CI |
| `src/ctrisk/ml/registry.py` | `models/v{N}/`: next version, save, load |
| `src/ctrisk/ml/train.py` | Fit baseline, main model, ablations; report; Snowflake clone + save |
| `src/ctrisk/ml/score.py` | Score active trials, top-3 drivers, append to `TRIAL_RISK_SCORES` |
| `tests/test_ml_features.py`, `tests/test_ml_evaluate.py`, `tests/test_ml_registry.py`, `tests/test_ml_train.py`, `tests/test_ml_score.py` | Tests |
| `Makefile`, `pyproject.toml`, `.gitignore`, `README.md` | Wiring |

---

### Task 1: Honest time split

**Files:** Modify `sql/20_features/40_trial_features.sql`, `sql/90_checks.sql`, `tests/test_features_sql.py`

- [ ] **Step 1: Update the split test first** — in `tests/test_features_sql.py`, add two boundary trials to the `RAW_TRIALS` insert (after the `T6` row, keeping the closing `"""`):

```sql
        ('T7', DATE '2015-01-01', 0,    'Eps',   'OTHER'),     -- first day of the test years
        ('T8', DATE '2014-12-31', 0,    'Eps',   'OTHER')      -- last day of the training years
```

and replace `test_split_by_start_year` with:

```python
def test_split_by_start_year(db):
    splits = {t: features(db, t)["split"] for t in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")}
    assert splits == {"T1": "test", "T2": "recent", "T3": "score", "T4": "train",
                      "T5": "recent", "T6": "recent", "T7": "test", "T8": "train"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_features_sql.py -v`
Expected: `test_split_by_start_year` FAILS (T1 is `train`, T2 is `test` under the old split).

- [ ] **Step 3: Change the split** — in `sql/20_features/40_trial_features.sql` replace the `CASE ... AS split` expression with:

```sql
       CASE WHEN t.label IS NULL THEN 'score'
            WHEN t.start_date < DATE '2015-01-01' THEN 'train'
            WHEN t.start_date < DATE '2017-01-01' THEN 'test'      -- resolved years: headline metric
            ELSE 'recent' END                                      AS split,  -- 2017-2020: still censored
```

In `sql/90_checks.sql`, rename `-- check: labels only in train and test` to `-- check: every split except score has labels` (query unchanged).

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass.

---

### Task 2: Model inputs

**Files:** Create `src/ctrisk/ml/__init__.py` (empty), `src/ctrisk/ml/features.py`, `tests/test_ml_features.py`; modify `pyproject.toml`

- [ ] **Step 1: Add dependencies**

Run: `uv add lightgbm scikit-learn pandas "snowflake-connector-python[pandas]"`

- [ ] **Step 2: Write failing tests** — `tests/test_ml_features.py`

```python
import pandas as pd

from ctrisk.ml.features import inputs, to_matrix

FRAME = pd.DataFrame({
    "nct_id": ["A", "B", "C"],
    "label": [1, 0, None],
    "split": ["train", "train", "score"],
    "start_date": pd.to_datetime(["2010-01-01", "2011-01-01", "2024-01-01"]),
    "phase": ["PHASE2", None, "PHASE1"],
    "n_countries": [3, 1, None],
    "has_faers_history": [True, False, None],
    "faers_reports": [10, 0, 0],
})


def test_inputs_never_include_ids_labels_or_dates():
    assert inputs(FRAME) == ["phase", "n_countries", "has_faers_history", "faers_reports"]
    assert inputs(FRAME, drop=["faers_reports"]) == ["phase", "n_countries", "has_faers_history"]


def test_matrix_fills_missing_categories_and_makes_everything_else_numeric():
    X, categories = to_matrix(FRAME, inputs(FRAME))
    assert list(X["phase"].astype(str)) == ["PHASE2", "missing", "PHASE1"]
    assert categories == {"phase": ["PHASE1", "PHASE2", "missing"]}
    assert X["has_faers_history"].tolist()[:2] == [1.0, 0.0]
    assert X["n_countries"].isna().tolist() == [False, False, True]


def test_scoring_reuses_training_categories():
    _, categories = to_matrix(FRAME, ["phase"])
    new = pd.DataFrame({"phase": ["PHASE3"]})           # unseen at training time
    X, _ = to_matrix(new, ["phase"], categories)
    assert list(X["phase"].cat.categories) == ["PHASE1", "PHASE2", "missing"]
    assert X["phase"].isna().all()                      # unknown -> missing value, not a new code
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_ml_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ctrisk.ml'`

- [ ] **Step 4: Implement** — `src/ctrisk/ml/features.py`

```python
"""Which TRIAL_FEATURES columns the model sees, and how they become a matrix."""
import numpy as np
import pandas as pd

NOT_INPUTS = {"nct_id", "label", "split", "start_date"}   # start_date is for splitting only
CATEGORICAL = {"phase", "allocation", "intervention_model", "primary_purpose", "masking",
               "sponsor_class", "sex"}
FAERS = ["n_substances", "faers_reports", "faers_reports_12m", "faers_serious_share",
         "faers_death_share", "has_faers_history"]
BURDEN = ["n_countries", "us_only", "min_age_years", "max_age_years", "healthy_volunteers",
          "criteria_count", "criteria_chars"]


def inputs(frame: pd.DataFrame, drop=()) -> list[str]:
    return [c for c in frame.columns if c not in NOT_INPUTS and c not in set(drop)]


def to_matrix(frame: pd.DataFrame, columns: list[str],
              categories: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Categoricals get a 'missing' level and fixed levels; everything else becomes float."""
    X = pd.DataFrame(index=frame.index)
    learned = {}
    for c in columns:
        if c in CATEGORICAL:
            values = frame[c].astype(object).where(frame[c].notna(), "missing").astype(str)
            levels = categories[c] if categories else sorted(values.unique())
            X[c] = pd.Categorical(values, categories=levels)
            learned[c] = levels
        else:
            X[c] = frame[c].map(lambda v: np.nan if pd.isna(v) else float(v))  # bool/Decimal/NA -> float
    return X, learned
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest -q && uv run ruff check src tests`

---

### Task 3: Evaluation

**Files:** Create `src/ctrisk/ml/evaluate.py`, `tests/test_ml_evaluate.py`

- [ ] **Step 1: Write failing tests** — `tests/test_ml_evaluate.py`

```python
import numpy as np

from ctrisk.ml.evaluate import bootstrap_auc_ci, calibration, metrics, precision_at_top

Y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
PERFECT = np.array([0.9, 0.1, 0.8, 0.2, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1])


def test_precision_at_top_counts_terminations_among_highest_scores():
    assert precision_at_top(Y, PERFECT, frac=0.2) == 1.0     # top 2 are both terminated
    assert precision_at_top(Y, -PERFECT, frac=0.2) == 0.0


def test_metrics_on_a_perfect_ranking():
    m = metrics(Y, PERFECT)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert (m["n"], m["base_rate"]) == (10, 0.2)


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    score = y * 0.5 + rng.random(500)
    lo, hi = bootstrap_auc_ci(y, score, n=200, seed=0)
    assert (lo, hi) == bootstrap_auc_ci(y, score, n=200, seed=0)
    assert lo < metrics(y, score)["roc_auc"] < hi


def test_calibration_bins_cover_every_row():
    rows = calibration(Y, PERFECT, bins=2)
    assert sum(r["n"] for r in rows) == len(Y)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ml_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `src/ctrisk/ml/evaluate.py`

```python
"""Ranking and calibration metrics for a termination-risk score."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def precision_at_top(y, score, frac: float = 0.10) -> float:
    """Share of terminated trials among the highest-scored `frac` of trials."""
    k = max(1, round(len(score) * frac))
    return float(np.asarray(y)[np.argsort(-np.asarray(score))[:k]].mean())


def metrics(y, score) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    return {"n": int(len(y)),
            "base_rate": round(float(y.mean()), 4),
            "roc_auc": round(float(roc_auc_score(y, score)), 4),
            "pr_auc": round(float(average_precision_score(y, score)), 4),
            "precision_top_10pct": round(precision_at_top(y, score), 4),
            "brier": round(float(brier_score_loss(y, score)), 4)}


def bootstrap_auc_ci(y, score, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval for ROC AUC from resampling the evaluation set."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], score[i]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def calibration(y, score, bins: int = 10) -> list[dict]:
    """Mean predicted risk vs observed termination rate, by score quantile."""
    df = pd.DataFrame({"y": y, "p": score})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    grouped = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return [{"bin": int(b), "mean_predicted": round(r.mean_predicted, 4),
             "observed": round(r.observed, 4), "n": int(r.n)} for b, r in grouped.iterrows()]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q && uv run ruff check src tests`

---

### Task 4: Model registry

**Files:** Create `src/ctrisk/ml/registry.py`, `tests/test_ml_registry.py`; modify `.gitignore`

- [ ] **Step 1: Write failing tests** — `tests/test_ml_registry.py`

```python
from ctrisk.ml.registry import latest, load, next_version, save


def test_versions_count_up_from_one(tmp_path):
    assert next_version(tmp_path) == 1
    save(tmp_path, 1, {"model": "stub"}, metrics={"roc_auc": 0.7})
    save(tmp_path, 2, {"model": "stub2"}, metrics={"roc_auc": 0.71})
    assert (next_version(tmp_path), latest(tmp_path)) == (3, 2)


def test_round_trip(tmp_path):
    save(tmp_path, 1, {"model": "stub"}, metrics={"roc_auc": 0.7}, manifest={"clone": "TRIAL_FEATURES_V1"})
    model, docs = load(tmp_path, 1)
    assert model == {"model": "stub"}
    assert docs == {"metrics": {"roc_auc": 0.7}, "manifest": {"clone": "TRIAL_FEATURES_V1"}}
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_ml_registry.py -v` → `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `src/ctrisk/ml/registry.py`

```python
"""Model versions on disk: models/v{N}/ holds model.pkl plus one JSON file per document."""
import json
import re
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def _versions(root: Path) -> list[int]:
    return sorted(int(m[1]) for p in Path(root).glob("v*") if (m := re.fullmatch(r"v(\d+)", p.name)))


def next_version(root: Path = MODELS_DIR) -> int:
    return (_versions(root) or [0])[-1] + 1


def latest(root: Path = MODELS_DIR) -> int:
    if not _versions(root):
        raise FileNotFoundError(f"no model versions in {root}; run `make train` first")
    return _versions(root)[-1]


def save(root: Path, version: int, model, **docs: dict) -> Path:
    folder = Path(root) / f"v{version}"
    folder.mkdir(parents=True, exist_ok=False)          # never overwrite a version
    joblib.dump(model, folder / "model.pkl")
    for name, doc in docs.items():
        (folder / f"{name}.json").write_text(json.dumps(doc, indent=2, default=str))
    return folder


def load(root: Path, version: int):
    folder = Path(root) / f"v{version}"
    docs = {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}
    return joblib.load(folder / "model.pkl"), docs
```

- [ ] **Step 4:** `.gitignore` already ignores `models/*/model.pkl`; confirm, so metrics/features/manifest JSON are committed and the binary is not.

- [ ] **Step 5: Run** — `uv run pytest -q && uv run ruff check src tests`

---

### Task 5: Training and report

**Files:** Create `src/ctrisk/ml/train.py`, `tests/test_ml_train.py`

- [ ] **Step 1: Write failing test** — `tests/test_ml_train.py`

```python
import numpy as np
import pandas as pd
import pytest

from ctrisk.ml.train import run


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(0)
    n = 4000
    signal = rng.normal(size=n)                          # the only real driver
    frame = pd.DataFrame({
        "nct_id": [f"NCT{i:05d}" for i in range(n)],
        "split": rng.choice(["train", "test", "recent", "score"], n, p=[0.55, 0.2, 0.15, 0.1]),
        "start_date": pd.Timestamp("2012-01-01"),
        "phase": rng.choice(["PHASE1", "PHASE2", "PHASE3"], n),
        "n_countries": signal,
        "faers_reports": rng.poisson(3, n),              # noise
    })
    frame["label"] = np.where(frame["split"] == "score", np.nan,
                              (signal + rng.normal(scale=0.5, size=n) > 1).astype(float))
    return run(frame)


def test_reports_every_model_on_test_and_recent(result):
    _, report = result
    assert set(report["models"]) == {"logistic_regression", "lightgbm", "lightgbm_no_faers", "lightgbm_no_burden"}
    for m in report["models"].values():
        assert set(m) == {"test", "recent"}
    assert report["models"]["lightgbm"]["test"]["roc_auc"] > 0.85
    lo, hi = report["models"]["lightgbm"]["test"]["roc_auc_ci95"]
    assert lo < report["models"]["lightgbm"]["test"]["roc_auc"] < hi


def test_ablation_without_the_real_driver_collapses(result):
    _, report = result
    assert report["models"]["lightgbm_no_burden"]["test"]["roc_auc"] < 0.65   # n_countries is in BURDEN


def test_top_driver_is_the_real_signal(result):
    _, report = result
    assert report["top_drivers"][0]["feature"] == "n_countries"


def test_returns_the_full_model_with_its_inputs(result):
    model, report = result
    assert report["features"]["columns"] == ["phase", "n_countries", "faers_reports"]
    assert hasattr(model, "predict_proba")
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_ml_train.py -v` → `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `src/ctrisk/ml/train.py`

```python
"""Train on 2008-2014 starts, evaluate on resolved 2015-2016 and censored 2017-2020, save a version.

LightGBM is the model that gets scored; logistic regression is the baseline it must beat.
Two ablations answer: how much do FAERS features add (Q3), and does the model lean on
burden features whose record edits could leak outcome information?
"""
import subprocess
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ctrisk.ml.evaluate import bootstrap_auc_ci, calibration, metrics
from ctrisk.ml.features import BURDEN, CATEGORICAL, FAERS, inputs, to_matrix

LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
ABLATIONS = {"lightgbm": [], "lightgbm_no_faers": FAERS, "lightgbm_no_burden": BURDEN}


def logistic_regression(columns: list[str]):
    cats = [c for c in columns if c in CATEGORICAL]
    nums = [c for c in columns if c not in CATEGORICAL]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    return make_pipeline(prep, LogisticRegression(max_iter=2000))


def evaluate(model, X: pd.DataFrame, y: pd.Series, full: bool) -> dict:
    p = model.predict_proba(X)[:, 1]
    out = metrics(y, p)
    if full:
        out["roc_auc_ci95"] = bootstrap_auc_ci(y, p)
        out["calibration"] = calibration(y, p)
    return out


def run(frame: pd.DataFrame):
    columns = inputs(frame)
    X, categories = to_matrix(frame, columns)
    y = frame["label"]
    rows = {s: frame["split"] == s for s in ("train", "test", "recent")}

    models, report = {}, {"models": {}}
    candidates = {"logistic_regression": (logistic_regression(columns), columns)}
    for name, drop in ABLATIONS.items():
        cols = [c for c in columns if c not in drop]
        candidates[name] = (lgb.LGBMClassifier(**LGBM_PARAMS), cols)
    for name, (model, cols) in candidates.items():
        model.fit(X.loc[rows["train"], cols], y[rows["train"]])
        models[name] = model
        report["models"][name] = {
            "test": evaluate(model, X.loc[rows["test"], cols], y[rows["test"]], full=name == "lightgbm"),
            "recent": evaluate(model, X.loc[rows["recent"], cols], y[rows["recent"]], full=False)}

    final = models["lightgbm"]
    contrib = final.predict(X.loc[rows["test"], columns], pred_contrib=True)[:, :-1]  # last column is bias
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=columns).sort_values(ascending=False)
    report["top_drivers"] = [{"feature": f, "mean_abs_contribution": round(float(v), 4)}
                             for f, v in importance.head(10).items()]
    report["splits"] = {s: {"n": int(r.sum()), "base_rate": round(float(y[r].mean()), 4)} for s, r in rows.items()}
    report["features"] = {"columns": columns, "categories": categories}
    return final, report


def _git_version() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


if __name__ == "__main__":
    from ctrisk.config import load_config
    from ctrisk.ml.registry import MODELS_DIR, next_version, save
    from ctrisk.warehouse.snowflake import connect

    load_config()  # loads .env
    version = next_version(MODELS_DIR)
    clone = f"TRIAL_FEATURES_V{version}"
    with connect() as conn, conn.cursor() as cur:
        # Frozen training data. Replace is safe: a version number is only used once its folder is saved.
        cur.execute(f"CREATE OR REPLACE TABLE {clone} CLONE TRIAL_FEATURES")
        frame = cur.execute(f"SELECT * FROM {clone}").fetch_pandas_all()
    frame.columns = frame.columns.str.lower()

    model, report = run(frame)
    manifest = {"version": version, "snowflake_clone": clone, "git": _git_version(),
                "trained_at": datetime.now(timezone.utc).isoformat(), "params": LGBM_PARAMS,
                "train": "start 2008-2014", "test": "start 2015-2016", "recent": "start 2017-2020"}
    folder = save(MODELS_DIR, version, model, metrics={k: v for k, v in report.items() if k != "features"},
                  features=report["features"], manifest=manifest)

    print(f"saved {folder}  (training data frozen as {clone})")
    print(f"{'model':<22}{'test AUC':>10}{'recent AUC':>12}{'test PR AUC':>13}{'top-10% precision':>19}")
    for name, m in report["models"].items():
        print(f"{name:<22}{m['test']['roc_auc']:>10}{m['recent']['roc_auc']:>12}"
              f"{m['test']['pr_auc']:>13}{m['test']['precision_top_10pct']:>19}")
    print("lightgbm test AUC 95% CI:", report["models"]["lightgbm"]["test"]["roc_auc_ci95"])
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest -q && uv run ruff check src tests`. If the synthetic thresholds are flaky, fix the data (more rows), never loosen `> 0.85` below 0.8.

---

### Task 6: Scoring

**Files:** Create `src/ctrisk/ml/score.py`, `tests/test_ml_score.py`

- [ ] **Step 1: Write failing tests** — `tests/test_ml_score.py`

```python
import numpy as np

from ctrisk.ml.score import risk_deciles, top_drivers


def test_top_drivers_are_the_largest_contributions_toward_termination():
    contrib = np.array([[0.5, -0.9, 0.1, 0.3],
                        [-0.2, 0.4, 0.8, 0.0]])
    assert top_drivers(contrib, ["a", "b", "c", "d"], k=2) == [["a", "d"], ["c", "b"]]


def test_deciles_run_from_1_lowest_to_10_highest():
    d = risk_deciles(np.linspace(0, 1, 100))
    assert (d.min(), d.max()) == (1, 10)
    assert (d == 10).sum() == 10 and d[-1] == 10
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_ml_score.py -v` → `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `src/ctrisk/ml/score.py`

```python
"""Score active trials with the latest model; append to TRIAL_RISK_SCORES with top drivers."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def top_drivers(contrib: np.ndarray, columns: list[str], k: int = 3) -> list[list[str]]:
    """Per trial, the k features pushing its risk up the most."""
    return [[columns[i] for i in np.argsort(-row)[:k]] for row in contrib]


def risk_deciles(p: np.ndarray) -> np.ndarray:
    """10 = the riskiest tenth of scored trials."""
    return pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).to_numpy() + 1


if __name__ == "__main__":
    from snowflake.connector.pandas_tools import write_pandas

    from ctrisk.config import load_config
    from ctrisk.ml.features import to_matrix
    from ctrisk.ml.registry import MODELS_DIR, latest, load
    from ctrisk.warehouse.snowflake import connect

    load_config()
    version = latest(MODELS_DIR)
    model, docs = load(MODELS_DIR, version)
    columns, categories = docs["features"]["columns"], docs["features"]["categories"]

    with connect() as conn:
        frame = conn.cursor().execute("SELECT * FROM TRIAL_FEATURES WHERE split = 'score'").fetch_pandas_all()
        frame.columns = frame.columns.str.lower()
        X, _ = to_matrix(frame, columns, categories)
        p = model.predict_proba(X)[:, 1]
        drivers = top_drivers(model.predict(X, pred_contrib=True)[:, :-1], columns)
        scores = pd.DataFrame({
            "NCT_ID": frame["nct_id"], "MODEL_VERSION": f"v{version}", "RISK_SCORE": p.round(4),
            "RISK_DECILE": risk_deciles(p),
            "TOP_DRIVER_1": [d[0] for d in drivers], "TOP_DRIVER_2": [d[1] for d in drivers],
            "TOP_DRIVER_3": [d[2] for d in drivers],
            "SCORED_AT": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")})
        ok, _, rows, _ = write_pandas(conn, scores, "TRIAL_RISK_SCORES", auto_create_table=True)
    print(f"appended {rows} scores from v{version} to TRIAL_RISK_SCORES" if ok else "write failed")
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest -q && uv run ruff check src tests`

---

### Task 7: Wiring

**Files:** Modify `Makefile`, `README.md`

- [ ] **Step 1: Makefile** — add `train score` to `.PHONY` and append:

```make
train:
	uv run python -m ctrisk.ml.train

score:
	uv run python -m ctrisk.ml.score
```

- [ ] **Step 2: README `## Run`** — append `make train  # clone TRIAL_FEATURES, fit, save models/vN` and `make score  # append active-trial scores to TRIAL_RISK_SCORES`.

- [ ] **Step 3: Run** — `uv run pytest -q && uv run ruff check src tests`

---

### Task 8: Real run (coordinator)

Snowflake compute only (trial credits); no GCP cost.

- [ ] **Step 1: Rebuild features with the new split** — `make warehouse`. Expected: rows for `recent`, `score`, `test`, `train`; test termination rate ≈ 14.8%, recent ≈ 18.7%.
- [ ] **Step 2: Train** — `make train`. Record every number printed; open `models/v1/metrics.json` and check calibration and top drivers make sense (no identifier-like or date-like driver).
- [ ] **Step 3: Sanity against leakage** — if test AUC > 0.85, stop and investigate before believing it (suspect a leaking feature: look at the top driver and the no_burden ablation).
- [ ] **Step 4: Score** — `make score`, then in Snowflake: `SELECT RISK_DECILE, COUNT(*), AVG(RISK_SCORE) FROM TRIAL_RISK_SCORES GROUP BY 1 ORDER BY 1`.
- [ ] **Step 5: Record** — README `## Results`: headline test AUC with 95% CI, PR AUC and base rate, top-10% precision vs base rate, baseline comparison, ablations (FAERS lift = Q3 preliminary; burden sensitivity), recent-year AUC with caveat, top drivers. Update the progress dashboard. Tell Frank what the resume bullet can now truthfully say.
