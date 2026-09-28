"""Score active trials with the latest model; append to TRIAL_RISK_SCORES with top drivers."""
from datetime import UTC, datetime

import numpy as np
import pandas as pd


def top_drivers(contrib: np.ndarray, columns: list[str], k: int = 3) -> list[list[str]]:
    """Per trial, up to k features pushing its risk up the most; None where fewer push up."""
    return [[columns[i] if row[i] > 0 else None for i in np.argsort(-row)[:k]] for row in contrib]


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
        if frame.empty:
            raise SystemExit("no active trials to score (split = 'score' is empty)")
        X, _ = to_matrix(frame, columns, categories)
        p = model.predict_proba(X)[:, 1]
        drivers = top_drivers(model.predict(X, pred_contrib=True)[:, :-1], columns)
        scores = pd.DataFrame({
            "NCT_ID": frame["nct_id"], "MODEL_VERSION": f"v{version}", "RISK_SCORE": p.round(4),
            "RISK_DECILE": risk_deciles(p),
            "TOP_DRIVER_1": [d[0] for d in drivers], "TOP_DRIVER_2": [d[1] for d in drivers],
            "TOP_DRIVER_3": [d[2] for d in drivers],
            "SCORED_AT": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")})
        ok, _, rows, _ = write_pandas(conn, scores, "TRIAL_RISK_SCORES", auto_create_table=True)
    print(f"appended {rows} scores from v{version} to TRIAL_RISK_SCORES" if ok else "write failed")
