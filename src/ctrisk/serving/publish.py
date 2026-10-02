"""Publish the lookup: Snowflake unloads TRIAL_LOOKUP to GCS, then current.json points the app at it.

Order matters: the new lookup file lands first, and only then does current.json switch to it, so the app
never reads a pointer to a file that is not there yet. Run after `make lookup`.
"""
import json
import os
import subprocess
from datetime import UTC, datetime

from ctrisk.ml.features import POST_START_LEAKS

REPO_URL = "https://github.com/frankf1747/clinical_trial_risk_pipeline"


def summary(metrics: dict, manifest: dict, audit: dict | None, counts: dict, lookup_file: str) -> dict:
    """What the app's front page says about the model, taken from the version's saved files."""
    t = metrics["targets"]
    test = t["label"]["models"]["lightgbm"]["test"]
    out = {
        "model_version": f"v{manifest['version']}", "trained_at": manifest["trained_at"][:10],
        "git": manifest["git"], "repo_url": REPO_URL, "lookup_file": lookup_file,
        "published_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "test": {"roc_auc": test["roc_auc"], "roc_auc_ci95": test.get("roc_auc_ci95"), "n": test["n"],
                 "base_rate": test["base_rate"], "precision_top_10pct": test["precision_top_10pct"],
                 "lift_top_10pct": round(test["precision_top_10pct"] / test["base_rate"], 1)},
        "enrollment_auc": t["label_enrollment"]["models"]["lightgbm"]["test"]["roc_auc"],
        "trials": counts,
        "left_out": list(POST_START_LEAKS),
        "audit": None,
    }
    if audit:
        m = audit["matched"]
        lo, hi = m["auc_drop_ci95"]
        out["audit"] = {"auc_change": round(-m["auc_drop"], 4), "ci95": [round(-hi, 4), round(-lo, 4)],
                        "n": m["n"], "archives": len(audit.get("archives") or [])}
    return out


if __name__ == "__main__":
    from pathlib import Path

    from ctrisk.config import load_config
    from ctrisk.ml.registry import MODELS_DIR, latest, load
    from ctrisk.warehouse.snowflake import connect
    from ctrisk.warehouse.sql import SQL_DIR, run_files

    load_config()
    bucket = os.environ["GCP_BUCKET"]
    version = latest(MODELS_DIR)
    _, docs = load(MODELS_DIR, version)
    lookup_file = f"lookup_v{version}.parquet"
    with connect() as conn:
        cur = conn.cursor()
        run_files(cur, [SQL_DIR / "40_lookup.sql"])
        counts = dict(cur.execute("SELECT score_type, COUNT(*) FROM TRIAL_LOOKUP GROUP BY 1").fetchall())
        if sum(counts.values()) == 0:
            raise SystemExit("TRIAL_LOOKUP is empty; run `make lookup` first")
        cur.execute(f"COPY INTO @SERVING_STAGE/{lookup_file} FROM (SELECT * FROM TRIAL_LOOKUP) "
                    "FILE_FORMAT = (TYPE = PARQUET) HEADER = TRUE SINGLE = TRUE OVERWRITE = TRUE "
                    "MAX_FILE_SIZE = 268435456")
        unloaded = cur.fetchall()
    doc = summary(docs["metrics"], docs["manifest"], docs.get("audit_point_in_time"), counts, lookup_file)
    local = Path("data") / "serving" / "current.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(doc, indent=2))
    subprocess.run(["gcloud", "storage", "cp", str(local), f"gs://{bucket}/serving/current.json"], check=True)
    print(f"published v{version}: {sum(counts.values()):,} trials {counts}; unload {unloaded}")
