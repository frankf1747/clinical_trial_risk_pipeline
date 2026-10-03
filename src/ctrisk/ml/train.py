"""Train on 2008-2014 starts, evaluate on resolved 2015-2016 and censored 2017-2020, save a version.

Three targets: any termination (scored as RISK_SCORE), termination for enrollment (scored as
ENROLLMENT_RISK_SCORE), and termination for safety (reported only: it is where FAERS should matter).
For each, logistic regression is the baseline; LightGBM is tuned on an inner time split; ablations
show what text, FAERS, and burden features contribute.
"""
import subprocess
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ctrisk.ml.embed import EmbeddingFeatures
from ctrisk.ml.evaluate import (
    bootstrap_auc_ci,
    calibration,
    calibration_fit,
    metrics,
    subgroup_auc,
)
from ctrisk.ml.features import (
    BURDEN,
    CATEGORICAL,
    EDIT_PRONE,
    FAERS,
    POST_START_LEAKS,
    inputs,
    to_matrix,
)
from ctrisk.ml.model import RiskModel, collapse_text
from ctrisk.ml.text import TextFeatures

TARGETS = {"label": "any termination", "label_enrollment": "terminated for enrollment",
           "label_safety": "terminated for safety"}
SCORED = ("label", "label_enrollment")
BASE_PARAMS = {"n_estimators": 2000, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
               "random_state": 0, "verbose": -1, "metric": "auc"}
GRID = [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
        for nl in (15, 31, 63) for lr in (0.03, 0.06) for mcs in (50, 200)]
TUNE_SPLIT = pd.Timestamp("2013-01-01")     # inner split: fit before, validate 2013-2014
ABLATIONS = {"lightgbm": ([], True), "lightgbm_no_text": ([], False),
             "lightgbm_no_faers": (FAERS, True), "lightgbm_no_burden": (BURDEN, True),
             "lightgbm_stable_only": (EDIT_PRONE, False)}    # no edit-prone fields, no text
# Rolling origin: for each year Y, fit on training-split starts before Y and test on starts in Y and Y+1.
# 2015 reproduces the headline split. Parameters stay as tuned (on 2013-2014), so the 2012 and 2013
# windows are slightly optimistic; text features are refit on each fold's training rows.
ROLLING = (2012, 2013, 2014, 2015)


def logistic_regression(columns: list[str]):
    cats = [c for c in columns if c in CATEGORICAL]
    nums = [c for c in columns if c not in CATEGORICAL]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    return make_pipeline(prep, LogisticRegression(max_iter=2000))


def _embed(rows: pd.DataFrame) -> EmbeddingFeatures | None:
    """Embedding components fit on these rows, when the frame carries embeddings (M9); else None."""
    return EmbeddingFeatures().fit(rows["embedding"]) if "embedding" in rows.columns else None


def tune(frame: pd.DataFrame, y: pd.Series, columns: list[str], grid: list[dict], text_min_df: int) -> tuple[dict, list]:
    """Pick LightGBM params on an inner time split of the training rows; text is fit once on the inner train."""
    inner = pd.to_datetime(frame["start_date"]) < TUNE_SPLIT
    text = TextFeatures(min_df=text_min_df).fit(frame.loc[inner, "text"])
    embed = _embed(frame[inner])
    results = []
    for g in grid:
        m = RiskModel(columns, {**BASE_PARAMS, **g}, text=text, fit_text=False, embed=embed, fit_embed=False)
        m.fit(frame[inner], y[inner], valid=(frame[~inner], y[~inner]))
        results.append({**g, "n_estimators": int(m.best_iteration),
                        "val_auc": round(float(roc_auc_score(y[~inner], m.predict_proba(frame[~inner]))), 4)})
    best = max(results, key=lambda r: r["val_auc"])
    params = {**BASE_PARAMS, **{k: best[k] for k in ("num_leaves", "learning_rate", "min_child_samples")},
              "n_estimators": max(50, best["n_estimators"])}
    return params, results


def rolling_origin(frame: pd.DataFrame, y: pd.Series, columns: list[str], params: dict, text_min_df: int,
                   years=ROLLING) -> list[dict]:
    """Held-out AUC and calibration on successive two-year windows: is the headline a lucky period?"""
    labeled = frame[frame["split"].isin(["train", "test"]) & y.notna()]
    year = pd.to_datetime(labeled["start_date"]).dt.year
    out = []
    for first in years:
        fit = labeled[(labeled["split"] == "train") & (year < first)]
        held = labeled[(year >= first) & (year < first + 2)]
        yf, yh = y[fit.index], y[held.index]
        if yf.nunique() < 2 or yh.nunique() < 2:
            continue
        embed = EmbeddingFeatures() if "embedding" in frame.columns else None
        p = RiskModel(columns, params, text=TextFeatures(min_df=text_min_df), embed=embed).fit(fit, yf).predict_proba(held)
        out.append({"test_years": f"{first}-{first + 1}", "train_n": len(fit), "n": len(held),
                    "positives": int(yh.sum()), "roc_auc": round(float(roc_auc_score(yh, p)), 4),
                    "calibration_fit": calibration_fit(yh, p)})
    return out


def evaluate(p: np.ndarray, y: pd.Series, full: bool) -> dict:
    out = metrics(y, p)
    if full:
        out["roc_auc_ci95"] = bootstrap_auc_ci(y, p)
        out["calibration"] = calibration(y, p)
        out["calibration_fit"] = calibration_fit(y, p)
    return out


def train_target(frame: pd.DataFrame, target: str, columns: list[str], grid: list[dict], text_min_df: int):
    y = frame[target]
    rows = {s: (frame["split"] == s) & y.notna() for s in ("train", "test", "recent")}
    train, test, recent = (frame[rows[s]] for s in ("train", "test", "recent"))
    params, results = tune(train, y[rows["train"]], columns, grid, text_min_df)

    models, report = {}, {"description": TARGETS[target], "params": params, "grid": results, "models": {},
                          "splits": {s: {"n": int(r.sum()), "base_rate": round(float(y[r].mean()), 4)}
                                     for s, r in rows.items()}}
    X_train, cats = to_matrix(train, columns)
    lr = logistic_regression(columns).fit(X_train, y[rows["train"]])
    models["logistic_regression"] = lambda f: lr.predict_proba(to_matrix(f, columns, cats)[0])[:, 1]
    shared = TextFeatures(min_df=text_min_df).fit(train["text"])
    shared_embed = _embed(train)
    ablations = dict(ABLATIONS)
    if shared_embed:                     # M9: which kind of text carries the signal
        ablations |= {"lightgbm_tfidf_only": ([], "tfidf"), "lightgbm_embeddings_only": ([], "embeddings")}
    for name, (drop, use_text) in ablations.items():
        cols = [c for c in columns if c not in drop]
        text = shared if use_text in (True, "tfidf") else None
        embed = shared_embed if use_text in (True, "embeddings") else None
        m = RiskModel(cols, params, text=text, fit_text=False, embed=embed, fit_embed=False).fit(train, y[rows["train"]])
        models[name] = m
    for name, m in models.items():
        predict = m if callable(m) else m.predict_proba     # the LR baseline is a plain function
        report["models"][name] = {
            "test": evaluate(predict(test), y[rows["test"]], full=name == "lightgbm"),
            "recent": evaluate(predict(recent), y[rows["recent"]], full=False)}
    p = pd.Series(models["lightgbm"].predict_proba(test), index=test.index)
    yt = y[rows["test"]]
    for key, col in (("by_sponsor_class", "sponsor_class"), ("by_phase", "phase")):
        if col in columns:
            report[key] = subgroup_auc(yt, p, test[col])
    report["by_start_year"] = subgroup_auc(yt, p, pd.to_datetime(test["start_date"]).dt.year)
    if target == "label":
        report["rolling_origin"] = rolling_origin(frame, y, columns, params, text_min_df)
    return models["lightgbm"], report


def _mean(values: pd.Series) -> float | None:
    return None if values.empty or values.isna().all() else round(float(values.mean()), 4)


def _is_boolean(values: pd.Series) -> bool:
    present = values.dropna()
    return present.empty or present.map(lambda v: isinstance(v, (bool, np.bool_))).all()


def driver_directions(contrib: np.ndarray, names: list[str], frame: pd.DataFrame) -> dict[str, dict | None]:
    """Which way each feature pushes, as mean signed contribution (log-odds; > 0 = toward termination):
    per level for categoricals and booleans, for the lowest and highest third of values for numerics.
    None for a composite with no single value (registration_text). Rows of contrib follow frame."""
    out = {}
    for j, name in enumerate(names):
        if name not in frame.columns:
            out[name] = None
            continue
        c, v = pd.Series(contrib[:, j], index=frame.index), frame[name]
        if name in CATEGORICAL or _is_boolean(v):
            levels = v.astype(object).where(v.notna(), "missing").astype(str)
            out[name] = {lvl: _mean(c[levels == lvl]) for lvl in sorted(levels.unique())}
        else:
            x = v.map(lambda a: np.nan if pd.isna(a) else float(a))
            lo, hi = x.quantile([1 / 3, 2 / 3])
            out[name] = {"low_third": _mean(c[x <= lo]), "high_third": _mean(c[x >= hi])}
    return out


def run(frame: pd.DataFrame, grid: list[dict] = GRID, text_min_df: int = 20, exclude=tuple(POST_START_LEAKS)):
    columns = inputs(frame, drop=exclude)
    labeled = frame["split"] != "score"
    if frame.loc[labeled, "label"].isna().any():
        raise ValueError("a train/test/recent row without a label; only score rows may be unlabeled")
    for col in ("label_enrollment", "label_safety"):
        if col in frame.columns and ((frame["label"] == 0) & frame[col].isna()).any():
            raise ValueError("reason labels must be 0 for completed trials")

    final, report = {}, {"targets": {}}
    for target in TARGETS:
        model, target_report = train_target(frame, target, columns, grid, text_min_df)
        report["targets"][target] = target_report
        if target in SCORED:
            final[target] = model

    main = final["label"]
    test = frame[(frame["split"] == "test") & frame["label"].notna()]
    contrib, names = collapse_text(main.contributions(test), main.feature_names)
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=names)       # size only; direction below
    directions = driver_directions(contrib, names, test)
    # registration_text sums 64 SVD columns, so its size is not comparable one-to-one with a single feature.
    report["top_drivers"] = [{"feature": f, "mean_abs_contribution": round(float(v), 4),
                              "direction": directions[f]}
                             for f, v in importance.sort_values(ascending=False).head(15).items()]
    report["features"] = {"columns": columns, "excluded": [c for c in exclude if c in frame.columns],
                          "categories": main.categories,
                          "text_components": len(main.text.columns) if main.text else 0}
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
    from pathlib import Path

    from ctrisk.config import load_config
    from ctrisk.ml.embed import attach
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
    frame["start_date"] = pd.to_datetime(frame["start_date"])
    embeddings = Path("data", "parquet", "trial_embeddings")
    if embeddings.exists():              # M9: biomedical sentence embeddings of the registration text
        frame = attach(frame, embeddings)

    git = _git_version()  # before training, so a git problem cannot cost a finished model
    models, report = run(frame)
    manifest = {"version": version, "snowflake_clone": clone, "git": git,
                "trained_at": datetime.now(UTC).isoformat(),
                "params": {t: report["targets"][t]["params"] for t in TARGETS},
                "text": "tf-idf svd" + (" + pubmedbert embeddings" if "embedding" in frame.columns else ""),
                "train": "start 2008-2014 (tuned on <2013 vs 2013-2014)", "test": "start 2015-2016",
                "recent": "start 2017-2020"}
    folder = save(MODELS_DIR, version, models, metrics={k: v for k, v in report.items() if k != "features"},
                  features=report["features"], manifest=manifest)

    print(f"saved {folder}  (training data frozen as {clone})")
    for target, t in report["targets"].items():
        print(f"\n== {target}: {t['description']}  (train n={t['splits']['train']['n']}, "
              f"base rate {t['splits']['train']['base_rate']})")
        print(f"{'model':<22}{'test AUC':>10}{'recent AUC':>12}{'test PR AUC':>13}{'top-10% precision':>19}")
        for name, m in t["models"].items():
            print(f"{name:<22}{m['test']['roc_auc']:>10}{m['recent']['roc_auc']:>12}"
                  f"{m['test']['pr_auc']:>13}{m['test']['precision_top_10pct']:>19}")
        print("lightgbm test AUC 95% CI:", t["models"]["lightgbm"]["test"]["roc_auc_ci95"],
              " calibration:", t["models"]["lightgbm"]["test"]["calibration_fit"])
        if any(m["test"]["roc_auc"] > 0.85 for m in t["models"].values()):
            print("!! a test AUC above 0.85 — investigate for leakage before believing it")
    for fold in report["targets"]["label"].get("rolling_origin", []):
        print(f"rolling origin, test starts {fold['test_years']}: AUC {fold['roc_auc']} "
              f"(n={fold['n']}, trained on {fold['train_n']}), calibration {fold['calibration_fit']}")
    for key in ("by_sponsor_class", "by_phase", "by_start_year"):
        if key in report["targets"]["label"]:
            print(f"\nlabel AUC {key.replace('_', ' ')}:", report["targets"]["label"][key])
    print("\nmain model contributors (mean |contribution|, direction):")
    for d in report["top_drivers"]:
        print(f"  {d['feature']:<34}{d['mean_abs_contribution']:>8}  {d['direction']}")
