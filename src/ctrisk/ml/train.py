"""Train on 2008-2014 starts, evaluate on resolved 2015-2016 and censored 2017-2020, save a version.

LightGBM is the model that gets scored; logistic regression is the baseline it must beat.
Two ablations answer: how much do FAERS features add (Q3), and does the model lean on
burden features whose record edits could leak outcome information?
"""
import subprocess
from datetime import UTC, datetime

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

LGBM_PARAMS = {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50,
               "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": 0, "verbose": -1}
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
    if any(y[r].isna().any() for r in rows.values()):
        raise ValueError("a train/test/recent row without a label; only score rows may be unlabeled")

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


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def _git_version() -> str:
    try:
        sha, dirty = _git("rev-parse", "--short", "HEAD"), _git("status", "--porcelain")
    except OSError:  # git not installed
        return "unknown"
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

    git = _git_version()  # before training, so a git problem cannot cost a finished model
    model, report = run(frame)
    manifest = {"version": version, "snowflake_clone": clone, "git": git,
                "trained_at": datetime.now(UTC).isoformat(), "params": LGBM_PARAMS,
                "train": "start 2008-2014", "test": "start 2015-2016", "recent": "start 2017-2020"}
    folder = save(MODELS_DIR, version, model, metrics={k: v for k, v in report.items() if k != "features"},
                  features=report["features"], manifest=manifest)

    print(f"saved {folder}  (training data frozen as {clone})")
    print(f"{'model':<22}{'test AUC':>10}{'recent AUC':>12}{'test PR AUC':>13}{'top-10% precision':>19}")
    for name, m in report["models"].items():
        print(f"{name:<22}{m['test']['roc_auc']:>10}{m['recent']['roc_auc']:>12}"
              f"{m['test']['pr_auc']:>13}{m['test']['precision_top_10pct']:>19}")
    print("lightgbm test AUC 95% CI:", report["models"]["lightgbm"]["test"]["roc_auc_ci95"])
