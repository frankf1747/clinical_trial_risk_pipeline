"""Prospective check: how well did scores written earlier rank the trials that have finished since?

`make score` appends each version's scores for active trials to TRIAL_RISK_SCORES. When a later AACT
snapshot is loaded (`make warehouse`), trials that have finished since carry a label in TRIAL_FEATURES.
Joining the two is a true forward test: nothing about those outcomes existed when the scores were
written. Caveat, stated with every result: the trials resolved so far over-represent early
terminations (they finish sooner), so precision reads high until the cohort has mostly resolved.
Re-run yearly.
"""
import json
from datetime import UTC, datetime

import pandas as pd

from ctrisk.ml.evaluate import bootstrap_auc_ci, calibration_fit, metrics

QUERY = """
SELECT s.nct_id, s.model_version, s.scored_at, s.risk_score, t.label
FROM TRIAL_RISK_SCORES s JOIN TRIAL_FEATURES t ON t.nct_id = s.nct_id
WHERE t.label IS NOT NULL"""

UNRESOLVED = """
SELECT model_version, COUNT(DISTINCT nct_id) FROM TRIAL_RISK_SCORES
WHERE nct_id NOT IN (SELECT nct_id FROM TRIAL_FEATURES WHERE label IS NOT NULL)
GROUP BY model_version"""


def summarize(resolved: pd.DataFrame, bootstrap: int = 1000) -> dict:
    """Per model version, metrics on each trial's first score from that version."""
    first = (resolved.sort_values("scored_at")
             .drop_duplicates(["nct_id", "model_version"], keep="first"))
    out = {}
    for version, g in first.groupby("model_version"):
        y, p = g["label"].to_numpy(dtype=float), g["risk_score"].to_numpy(dtype=float)
        if y.min() == y.max():
            out[version] = {"n": len(g), "note": "no terminated trials resolved yet" if y.max() == 0
                            else "no completed trials resolved yet"}
            continue
        out[version] = {**metrics(y, p), "roc_auc_ci95": bootstrap_auc_ci(y, p, n=bootstrap),
                        "calibration_fit": calibration_fit(y, p),
                        "first_scored": pd.Timestamp(g["scored_at"].min()).strftime("%Y-%m-%d")}
    return out


if __name__ == "__main__":
    from ctrisk.config import load_config
    from ctrisk.ml.registry import MODELS_DIR
    from ctrisk.warehouse.snowflake import connect

    load_config()
    with connect() as conn:
        resolved = conn.cursor().execute(QUERY).fetch_pandas_all()
        unresolved = dict(conn.cursor().execute(UNRESOLVED).fetchall())
    resolved.columns = resolved.columns.str.lower()
    report = {"run_at": datetime.now(UTC).isoformat(), "unresolved_trials": unresolved,
              "by_version": summarize(resolved) if len(resolved) else {}}
    out = MODELS_DIR / f"backtest_{datetime.now(UTC):%Y-%m-%d}.json"
    out.write_text(json.dumps(report, indent=2))
    print("scored trials still running:", unresolved)
    for version, m in report["by_version"].items():
        print(f"{version}: {m}")
    print(f"wrote {out}  (resolved trials skew toward early terminations until most have finished)")
