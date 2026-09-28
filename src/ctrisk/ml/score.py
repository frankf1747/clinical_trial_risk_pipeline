"""Score active trials with the latest version: overall and enrollment termination risk, plus top drivers."""
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ctrisk.ml.model import collapse_text


def top_drivers(contrib: np.ndarray, columns: list[str], k: int = 3) -> list[list[str]]:
    """Per trial, up to k features pushing its risk up the most; None where fewer push up."""
    return [[columns[i] if row[i] > 0 else None for i in np.argsort(-row)[:k]] for row in contrib]


def risk_deciles(p: np.ndarray) -> np.ndarray:
    """10 = the riskiest tenth of scored trials."""
    return pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).to_numpy() + 1


if __name__ == "__main__":
    from snowflake.connector.pandas_tools import write_pandas

    from ctrisk.config import load_config
    from ctrisk.ml.registry import MODELS_DIR, latest, load
    from ctrisk.warehouse.snowflake import connect

    load_config()
    version = latest(MODELS_DIR)
    models, _ = load(MODELS_DIR, version)
    if not isinstance(models, dict):
        raise SystemExit(f"v{version} predates M5; run `make train` first")
    overall, enrollment = models["label"], models["label_enrollment"]

    with connect() as conn:
        frame = conn.cursor().execute("SELECT * FROM TRIAL_FEATURES WHERE split = 'score'").fetch_pandas_all()
        frame.columns = frame.columns.str.lower()
        if frame.empty:
            raise SystemExit("no active trials to score (split = 'score' is empty)")
        p = overall.predict_proba(frame)
        contrib, names = collapse_text(overall.contributions(frame), overall.feature_names)
        drivers = top_drivers(contrib, names)
        scores = pd.DataFrame({
            "NCT_ID": frame["nct_id"], "MODEL_VERSION": f"v{version}",
            "RISK_SCORE": p.round(4), "RISK_DECILE": risk_deciles(p),
            "ENROLLMENT_RISK_SCORE": enrollment.predict_proba(frame).round(4),
            "TOP_DRIVER_1": [d[0] for d in drivers], "TOP_DRIVER_2": [d[1] for d in drivers],
            "TOP_DRIVER_3": [d[2] for d in drivers],
            "SCORED_AT": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")})
        conn.cursor().execute(
            "ALTER TABLE IF EXISTS TRIAL_RISK_SCORES ADD COLUMN IF NOT EXISTS ENROLLMENT_RISK_SCORE FLOAT")
        ok, _, rows, _ = write_pandas(conn, scores, "TRIAL_RISK_SCORES", auto_create_table=True)
    print(f"appended {rows} scores from v{version} to TRIAL_RISK_SCORES" if ok else "write failed")
