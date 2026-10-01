"""Does the held-out AUC survive when registry features come from the record as it stood at trial start?

Scores the saved model twice on the same 2017-2020 trials: once on TRIAL_FEATURES (latest record), once
with the registry-derived columns replaced by their values in the AACT archive nearest each start
(src/ctrisk/spark/point_in_time.py). FAERS and sponsor-history columns are already point-in-time and
stay as they are. Reported:

- AUC and calibration both ways, on the trials that have an archived record, with a paired bootstrap
  interval on the drop; and on the strict subset whose record predates the start (lag_days <= 0).
- For each registry column, the AUC drop when only that column is swapped, and how often it changed
  between start and now for terminated vs completed trials. A column that changed more often for
  terminated trials carries the outcome back in time. A column that changed for everyone, or whose
  archived values the model never saw (unseen_levels), points at a format change, not at leakage.
"""
import json
import numbers

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ctrisk.ml.evaluate import bootstrap_auc_ci, calibration, calibration_fit, metrics

REGISTRY = ["phase", "number_of_arms", "allocation", "intervention_model", "primary_purpose", "masking",
            "sponsor_class", "has_dmc", "n_countries", "us_only", "min_age_years", "max_age_years",
            "healthy_volunteers", "sex", "criteria_count", "criteria_chars", "responsible_party",
            "n_collaborators", "n_keywords", "text"]


def registry_columns(columns) -> list[str]:
    """The registry-derived columns present in a frame, in REGISTRY order, then the disease areas."""
    present = list(columns)
    return [c for c in REGISTRY if c in present] + [c for c in present if c.startswith("area_")]


def fill_unarchived(pit: pd.DataFrame, latest: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """Where an archive could not record a column (null disease areas from archives without MeSH
    ancestors), keep the latest value. Returns the filled frame and the share filled per column."""
    out, share = pit.copy(), {}
    now = latest.set_index("nct_id")
    for c in columns:
        if c in out.columns and c in now.columns:
            missing = out[c].isna()
            out[c] = out[c].astype(object).where(~missing, out["nct_id"].map(now[c]))
            share[c] = round(float(missing.mean()), 4)
    return out, share


def overlay(latest: pd.DataFrame, pit: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Latest-record rows with `columns` replaced by their archived values; trials without one are dropped."""
    cols = [c for c in columns if c in pit.columns]
    out = latest.drop(columns=cols).merge(pit[["nct_id", *cols]], on="nct_id", how="inner")
    return out[latest.columns]


def _canon(values: pd.Series) -> pd.Series:
    """Comparable form: 3 == 3.0 == Decimal('3'), True == np.True_, None == NaN."""
    def one(v):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return "<NA>"
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, numbers.Number):
            return repr(float(v))
        return str(v)
    return values.map(one)


def change_rates(latest: pd.DataFrame, pit: pd.DataFrame, columns: list[str]) -> dict:
    both = latest.merge(pit, on="nct_id", suffixes=("", "_pit"))
    out = {}
    for c in columns:
        if c not in pit.columns:
            continue
        differs = _canon(both[c]) != _canon(both[f"{c}_pit"])
        t, k = float(differs[both["label"] == 1].mean()), float(differs[both["label"] == 0].mean())
        out[c] = {"terminated": round(t, 4), "completed": round(k, 4), "gap": round(t - k, 4)}
    return out


def unseen_levels(pit: pd.DataFrame, categories: dict) -> dict:
    """Share of archived values of each categorical that the model never saw in training."""
    out = {}
    for c, levels in categories.items():
        if c in pit.columns:
            present = pit[c].dropna().astype(str)
            share = float((~present.isin(levels)).mean()) if len(present) else 0.0
            if share:
                out[c] = round(share, 4)
    return out


def paired_auc_diff_ci(y, a, b, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval for AUC(a) - AUC(b) on the same trials, resampling trials once for both."""
    y, a, b = (np.asarray(v, dtype=float) for v in (y, a, b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def _both_ways(model, latest: pd.DataFrame, pit: pd.DataFrame, columns: list[str], bootstrap: int) -> dict:
    swapped = overlay(latest, pit, columns)
    same = latest.set_index("nct_id").loc[swapped["nct_id"]].reset_index()
    y = swapped["label"].to_numpy(dtype=float)
    a, b = model.predict_proba(same), model.predict_proba(swapped)
    out = {"n": len(y)}
    for name, p in (("latest", a), ("point_in_time", b)):
        out[name] = {**metrics(y, p), "roc_auc_ci95": bootstrap_auc_ci(y, p, n=bootstrap),
                     "calibration_fit": calibration_fit(y, p)}
    out["point_in_time"]["calibration"] = calibration(y, b)
    out["auc_drop"] = round(out["latest"]["roc_auc"] - out["point_in_time"]["roc_auc"], 4)
    out["auc_drop_ci95"] = paired_auc_diff_ci(y, a, b, n=bootstrap)
    return out


def compare(model, latest: pd.DataFrame, pit: pd.DataFrame, columns: list[str], bootstrap: int = 1000) -> dict:
    pit, areas_from_latest = fill_unarchived(pit, latest, [c for c in columns if c.startswith("area_")])
    matched = latest[latest["nct_id"].isin(pit["nct_id"])]
    y = matched["label"].to_numpy(dtype=float)
    base = roc_auc_score(y, model.predict_proba(matched))
    swap_one = {}
    for c in columns:
        if c in pit.columns:
            p = model.predict_proba(overlay(matched, pit, [c]).set_index("nct_id").loc[matched["nct_id"]].reset_index())
            swap_one[c] = round(float(base - roc_auc_score(y, p)), 4)
    strict = pit[pit["lag_days"] <= 0] if "lag_days" in pit.columns else pit
    return {
        "n_cohort": len(latest), "n_matched": len(matched),
        "latest_full_cohort_auc": round(float(roc_auc_score(latest["label"], model.predict_proba(latest))), 4),
        "matched": _both_ways(model, latest, pit, columns, bootstrap),
        "strict": _both_ways(model, latest, strict, columns, bootstrap),
        "lag_days": ({q: float(pit["lag_days"].quantile(v)) for q, v in (("p10", .1), ("median", .5), ("p90", .9))}
                     if "lag_days" in pit.columns else None),
        "swap_one_column": dict(sorted(swap_one.items(), key=lambda kv: -kv[1])),
        "change_rates": dict(sorted(change_rates(latest, pit, columns).items(), key=lambda kv: -kv[1]["gap"])),
        "unseen_levels": unseen_levels(pit, getattr(model, "categories", None) or {}),
        "areas_from_latest": areas_from_latest,
        "archives": (sorted(pd.to_datetime(pit["archive_date"]).dt.strftime("%Y-%m-%d").unique().tolist())
                     if "archive_date" in pit.columns else []),
    }


if __name__ == "__main__":
    from ctrisk.config import load_config
    from ctrisk.ml.registry import MODELS_DIR, latest, load
    from ctrisk.warehouse.snowflake import connect

    cfg = load_config()
    version = latest(MODELS_DIR)
    models, docs = load(MODELS_DIR, version)
    model, clone = models["label"], docs["manifest"]["snowflake_clone"]
    with connect() as conn:          # the frozen rows this version was evaluated on
        frame = conn.cursor().execute(f"SELECT * FROM {clone} WHERE split = 'recent'").fetch_pandas_all()
    frame.columns = frame.columns.str.lower()
    pit = pd.read_parquet(cfg.path("parquet_pit", "registry_at_start"))   # local parquet (MODE=local)
    report = compare(model, frame, pit, registry_columns(frame.columns))

    out = MODELS_DIR / f"v{version}" / "audit_point_in_time.json"
    out.write_text(json.dumps(report, indent=2))
    m, s = report["matched"], report["strict"]
    print(f"v{version}, 2017-2020 starts: {report['n_matched']} of {report['n_cohort']} trials have an archived record")
    for label, r in (("matched", m), ("record predates start", s)):
        print(f"  {label:<22} n={r['n']:<6} latest AUC {r['latest']['roc_auc']}  point-in-time AUC "
              f"{r['point_in_time']['roc_auc']}  drop {r['auc_drop']} (95% CI {r['auc_drop_ci95']})")
    print("  calibration (point-in-time):", m["point_in_time"]["calibration_fit"])
    print("\nAUC drop when only this column is point-in-time, and how often it changed since start:")
    for c, drop in list(report["swap_one_column"].items())[:12]:
        print(f"  {c:<22}{drop:>8}   changed: terminated {report['change_rates'][c]['terminated']:.1%}, "
              f"completed {report['change_rates'][c]['completed']:.1%}")
    if report["unseen_levels"]:
        print("\n!! archived values the model never saw (format drift, or an edit from outside the modelled population):", report["unseen_levels"])
    print(f"\nwrote {out}")
