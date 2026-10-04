"""Compare two model versions on the same held-out trials: paired bootstrap of the AUC difference.

Both versions score the test rows of the newer version's frozen clone, so the trials and labels are
identical; resampling the trials together keeps the comparison paired.
    python -m ctrisk.ml.compare 5 6     -> models/v6/compare_v5.json
"""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from ctrisk.ml.train import SCORED


def paired_auc_difference(y, a, b, n: int = 1000, seed: int = 0) -> dict:
    """AUC of a and b, and a 95% interval for AUC(b) - AUC(a) from resampling trials together."""
    y, a, b = (np.asarray(v, dtype=float) for v in (y, a, b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    auc_a, auc_b = roc_auc_score(y, a), roc_auc_score(y, b)
    return {"n": len(y), "positives": int(y.sum()), "auc_old": round(float(auc_a), 4), "auc_new": round(float(auc_b), 4),
            "difference": round(float(auc_b - auc_a), 4), "difference_ci95": [round(float(lo), 4), round(float(hi), 4)]}


if __name__ == "__main__":
    from ctrisk.config import load_config
    from ctrisk.ml.registry import MODELS_DIR, load
    from ctrisk.warehouse.snowflake import connect

    old, new = int(sys.argv[1]), int(sys.argv[2])
    load_config()
    (old_models, _), (new_models, new_docs) = load(MODELS_DIR, old), load(MODELS_DIR, new)
    with connect() as conn:
        frame = conn.cursor().execute(f"SELECT * FROM {new_docs['manifest']['snowflake_clone']} "
                                      "WHERE split = 'test' ORDER BY nct_id").fetch_pandas_all()
    frame.columns = frame.columns.str.lower()
    out = {}
    for target in SCORED:
        rows = frame[frame[target].notna()]
        out[target] = paired_auc_difference(rows[target], old_models[target].predict_proba(rows),
                                            new_models[target].predict_proba(rows))
        print(f"v{new} vs v{old}, {target}: {out[target]}")
    path = MODELS_DIR / f"v{new}" / f"compare_v{old}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
