"""Fit and evaluate the competing-risks survival model (M8) on the rows the yes/no model vN was built on.

Fit: 2008-2014 starts, finished (terminated or completed, with their durations) and still running
(censored at the snapshot). Tuning: fit on starts before 2013, validate on 2013-2014. Test: 2015-2016
starts (follow-up nearly complete) and 2017-2020 starts (heavily censored). Compared on the same trials
with vN's yes/no score. Saved to models/survival/sN/, apart from the yes/no versions.
"""
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ctrisk.ml.registry import MODELS_DIR, latest, load, next_version, save
from ctrisk.ml.survival import PERIODS, WIDTH, SurvivalModel, durations
from ctrisk.ml.survival_eval import (
    aalen_johansen,
    brier,
    calibration_by_decile,
    time_auc,
)
from ctrisk.ml.text import TextFeatures

SURVIVAL_DIR = MODELS_DIR / "survival"
HORIZONS = (1.0, 2.0, 3.0, 5.0)
FIT_END, TUNE_SPLIT = pd.Timestamp("2015-01-01"), pd.Timestamp("2013-01-01")
COHORTS = {"test": ("2015-01-01", "2017-01-01"), "recent": ("2017-01-01", "2021-01-01")}
BASE = {"n_estimators": 2000, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        "random_state": 0, "verbose": -1}
GRID = [{"num_leaves": nl, "learning_rate": 0.05, "min_child_samples": 200} for nl in (15, 31, 63)]


def population(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known duration or a censoring time, and their time/event."""
    d = durations(frame)
    out = frame.loc[d.index].copy()
    out[["time", "event"]] = d
    out["start"] = pd.to_datetime(out["start_date"])
    return out


def tune(fit: pd.DataFrame, columns: list[str], grid: list[dict], text_min_df: int) -> tuple[dict, list]:
    inner, valid = fit[fit["start"] < TUNE_SPLIT], fit[fit["start"] >= TUNE_SPLIT]
    results = []
    for g in grid:
        m = SurvivalModel(columns, {**BASE, **g}, text=TextFeatures(min_df=text_min_df))
        m.fit(inner, inner["time"], inner["event"], valid=(valid, valid["time"], valid["event"]))
        auc = time_auc(valid["time"], valid["event"], m.cif(valid, [2.0])[:, 0], 2.0)
        results.append({**g, "n_estimators": m.best_iteration, "valid_time_auc_2y": round(auc, 4)})
    best = max(results, key=lambda r: r["valid_time_auc_2y"])
    return {**BASE, **{k: best[k] for k in ("num_leaves", "learning_rate", "min_child_samples", "n_estimators")}}, results


def paired_boot(t, e, a, b, horizon: float, n: int = 200, seed: int = 0) -> dict:
    """95% intervals for time-AUC(a), and for time-AUC(a) - time-AUC(b) on the same resampled trials."""
    rng = np.random.default_rng(seed)
    t, e, a, b = map(np.asarray, (t, e, a, b))
    one, diff = [], []
    for _ in range(n):
        i = rng.integers(0, len(t), len(t))
        x = time_auc(t[i], e[i], a[i], horizon)
        one.append(x)
        diff.append(x - time_auc(t[i], e[i], b[i], horizon))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"ci95": q(one), "difference_ci95": q(diff)}


def evaluate(model: SurvivalModel, fit: pd.DataFrame, cohort: pd.DataFrame, yes_no: np.ndarray) -> dict:
    h = model.hazards(cohort)
    cif = model.cif(cohort, list(HORIZONS), h=h)
    t, e = cohort["time"].to_numpy(), cohort["event"].to_numpy()
    out = {"n": len(cohort), "terminated": int((e == 1).sum()), "completed": int((e == 2).sum()),
           "censored": int((e == 0).sum()), "by_horizon": {}}
    for j, hz in enumerate(HORIZONS):
        null = aalen_johansen(fit["time"], fit["event"], hz)            # same risk for everyone
        out["by_horizon"][f"{hz:g}y"] = {
            "observed_cif": round(aalen_johansen(t, e, hz), 4),
            "mean_predicted_cif": round(float(cif[:, j].mean()), 4),
            "time_auc": round(time_auc(t, e, cif[:, j], hz), 4),
            "time_auc_yes_no_model": round(time_auc(t, e, yes_no, hz), 4),
            "brier": round(brier(t, e, cif[:, j], hz), 4),
            "brier_null": round(brier(t, e, np.full(len(t), null), hz), 4),
        }
    out["time_auc_2y_bootstrap"] = paired_boot(t, e, cif[:, 1], yes_no, 2.0)
    out["calibration_2y"] = calibration_by_decile(t, e, cif[:, 1], 2.0)
    out["calibration_5y"] = calibration_by_decile(t, e, cif[:, 3], 5.0)
    return out


def run(frame: pd.DataFrame, columns: list[str], yes_no_model, grid=GRID, text_min_df: int = 20):
    pop = population(frame)
    fit = pop[pop["start"] < FIT_END]
    fit = fit[(fit["split"] == "train") | (fit["event"] == 0)]          # finished training trials + censored
    params, results = tune(fit, columns, grid, text_min_df)
    model = SurvivalModel(columns, params, text=TextFeatures(min_df=text_min_df)).fit(fit, fit["time"], fit["event"])
    report = {"params": params, "grid": results, "width_years": WIDTH, "periods": PERIODS,
              "fit": {"n": len(fit), "terminated": int((fit["event"] == 1).sum()),
                      "completed": int((fit["event"] == 2).sum()), "censored": int((fit["event"] == 0).sum()),
                      "dropped_no_end_date": int(frame["label"].notna().sum() - pop["label"].notna().sum())},
              "cohorts": {}}
    for name, (lo, hi) in COHORTS.items():
        c = pop[(pop["start"] >= lo) & (pop["start"] < hi)]
        report["cohorts"][name] = evaluate(model, fit, c, yes_no_model.predict_proba(c))
    return model, report


if __name__ == "__main__":
    from ctrisk.config import load_config
    from ctrisk.warehouse.snowflake import connect
    from ctrisk.warehouse.sql import SQL_DIR, run_files

    load_config()
    version = latest(MODELS_DIR)
    models, docs = load(MODELS_DIR, version)
    clone, columns = docs["manifest"]["snowflake_clone"], docs["features"]["columns"]
    with connect() as conn:
        run_files(conn.cursor(), [SQL_DIR / "50_outcomes.sql"])
        frame = conn.cursor().execute(f"SELECT f.*, o.end_date FROM {clone} f "
                                      "LEFT JOIN TRIAL_OUTCOMES o ON o.nct_id = f.nct_id").fetch_pandas_all()
    frame.columns = frame.columns.str.lower()
    model, report = run(frame.reset_index(drop=True), columns, models["label"])
    sv = next_version(SURVIVAL_DIR)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    manifest = {"version": sv, "yes_no_version": version, "snowflake_clone": clone, "git": git,
                "trained_at": datetime.now(UTC).isoformat(), "columns": columns}
    folder = save(SURVIVAL_DIR, sv, model, metrics=report, manifest=manifest)
    for name, r in report["cohorts"].items():
        print(f"== {name}: {r['n']} trials, {r['terminated']} terminated, {r['censored']} still running")
        for hz, m in r["by_horizon"].items():
            print(f"  {hz}: time-AUC {m['time_auc']} (yes/no model {m['time_auc_yes_no_model']}), "
                  f"Brier {m['brier']} vs null {m['brier_null']}, CIF predicted {m['mean_predicted_cif']} "
                  f"observed {m['observed_cif']}")
        print("  2y bootstrap:", r["time_auc_2y_bootstrap"])
    print(f"saved {folder}  ({Path(folder).name}, params {report['params']})")
    print(json.dumps(report["grid"]))
