"""Build the risk dashboard: Snowflake serving views + the latest model's metrics -> one HTML file.

The page is self-contained (data inlined), so it opens from disk, on GitHub Pages, or as a hosted artifact.
"""
import json
from pathlib import Path

import pandas as pd

TEMPLATE = Path(__file__).with_name("dashboard.html")
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs" / "dashboard" / "index.html"


def _target(t: dict) -> dict:
    lg, base = t["models"]["lightgbm"]["test"], t["models"]["logistic_regression"]["test"]
    return {"auc": lg["roc_auc"], "ci": list(lg.get("roc_auc_ci95") or []), "baseline": base["roc_auc"],
            "pr_auc": lg["pr_auc"], "top10": lg["precision_top_10pct"], "base_rate": lg["base_rate"],
            "n": lg["n"], "positives": round(lg["n"] * lg["base_rate"])}


def payload(active: pd.DataFrame, areas: pd.DataFrame, metrics: dict, manifest: dict, top_n: int = 400) -> dict:
    targets = metrics["targets"]
    label = targets["label"]["models"]
    full = label["lightgbm"]["test"]["roc_auc"]
    top = active.sort_values("risk_score", ascending=False).head(top_n)
    drivers = top[["top_driver_1", "top_driver_2", "top_driver_3"]].to_numpy()
    return {
        "summary": {
            "model_version": f"v{manifest['version']}", "git": manifest["git"],
            "trained_at": manifest["trained_at"][:10], "snowflake_clone": manifest["snowflake_clone"],
            "scored_at": str(active["scored_at"].max())[:10], "n_active": len(active), "n_listed": len(top),
            "targets": {name: _target(t) for name, t in targets.items()},
            "by_sponsor": [{"key": k, "auc": v["roc_auc"], "n": v["n"]}
                           for k, v in targets["label"].get("by_sponsor_class", {}).items()],
            "ablations": {k: round(full - label[f"lightgbm_no_{k}"]["test"]["roc_auc"], 4)
                          for k in ("text", "burden", "faers")},
            "calibration": label["lightgbm"]["test"]["calibration"],
            "top_drivers": metrics["top_drivers"],
        },
        "areas": [{"area": r.disease_area, "trials": int(r.trials), "avg_risk": round(float(r.avg_risk), 4),
                   "top_decile": int(r.top_decile_trials)} for r in areas.itertuples()],
        "trials": [{"nct": r.nct_id, "title": r.brief_title if pd.notna(r.brief_title) else None,
                    "phase": r.phase, "sponsor": r.sponsor_class, "area": r.disease_area,
                    "risk": round(float(r.risk_score), 4), "decile": int(r.risk_decile),
                    "enrollment_risk": round(float(r.enrollment_risk_score), 4),
                    "drivers": [d for d in ds if isinstance(d, str)]}
                   for r, ds in zip(top.itertuples(), drivers, strict=True)],
    }


def render(data: dict, standalone: bool = True) -> str:
    """Inline the data into the template. `</` is escaped so no value can close the script tag."""
    body = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    if not standalone:
        return body
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"</head>\n<body>\n{body}\n</body>\n</html>\n")


if __name__ == "__main__":
    from ctrisk.config import load_config
    from ctrisk.ml.registry import MODELS_DIR, latest
    from ctrisk.warehouse.snowflake import connect
    from ctrisk.warehouse.sql import SQL_DIR, run_files

    load_config()
    folder = MODELS_DIR / f"v{latest(MODELS_DIR)}"
    metrics, manifest = (json.loads((folder / f"{n}.json").read_text()) for n in ("metrics", "manifest"))
    with connect() as conn, conn.cursor() as cur:
        run_files(cur, [SQL_DIR / "30_serving.sql"])
        active, areas = (cur.execute(f"SELECT * FROM {v}").fetch_pandas_all() for v in ("VW_ACTIVE_TRIAL_RISK", "VW_RISK_BY_AREA"))
    for frame in (active, areas):
        frame.columns = frame.columns.str.lower()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(payload(active, areas, metrics, manifest)))
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1e3:.0f} KB, {len(active):,} active trials)")
