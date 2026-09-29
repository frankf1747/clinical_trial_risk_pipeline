"""Score active trials with the latest version: overall and enrollment termination risk, plus the
top model contributors per trial as (feature=value, signed contribution).

Contributors are LightGBM SHAP contributions in log-odds: what moved this trial's score, not why
the trial would stop.
"""
import numbers
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ctrisk.ml.model import collapse_text

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
    None where fewer than k push up. Rows of `contrib` line up with rows of `frame` by position."""
    values = frame.reindex(columns=columns).to_dict("records")     # composites come back as NaN
    out = []
    for row, trial in zip(contrib, values):
        out.append([(describe(columns[i], trial[columns[i]]), round(float(row[i]), 4)) if row[i] > 0 else None
                    for i in np.argsort(-row)[:k]])
    return out


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
        # Append-only table: earlier versions' rows keep bare feature names and no contribution.
        for col in ("ENROLLMENT_RISK_SCORE", "TOP_DRIVER_1_CONTRIB", "TOP_DRIVER_2_CONTRIB", "TOP_DRIVER_3_CONTRIB"):
            conn.cursor().execute(f"ALTER TABLE IF EXISTS TRIAL_RISK_SCORES ADD COLUMN IF NOT EXISTS {col} FLOAT")
        ok, _, rows, _ = write_pandas(conn, scores, "TRIAL_RISK_SCORES", auto_create_table=True)
    print(f"appended {rows} scores from v{version} to TRIAL_RISK_SCORES" if ok else "write failed")
