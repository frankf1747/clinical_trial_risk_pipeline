import numpy as np
import pandas as pd

from ctrisk.ml.backtest import summarize


def scores(seed=0, n=600):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    risk = np.clip(0.2 + 0.3 * label + rng.normal(scale=0.2, size=n), 0.01, 0.99)
    return pd.DataFrame({"nct_id": [f"T{i}" for i in range(n)], "model_version": "v3",
                         "scored_at": pd.Timestamp("2026-10-01"), "risk_score": risk, "label": label})


def test_summarize_ranks_each_version_on_its_first_score_per_trial():
    first = scores()
    rescored = first.assign(scored_at=pd.Timestamp("2027-01-01"), risk_score=0.5)   # later, uninformative
    out = summarize(pd.concat([first, rescored]), bootstrap=100)
    v3 = out["v3"]
    assert v3["n"] == 600 and v3["roc_auc"] > 0.8               # the later constant scores were dropped
    assert v3["first_scored"] == "2026-10-01" and set(v3["calibration_fit"]) == {"intercept", "slope"}


def test_summarize_skips_versions_without_both_outcomes_yet():
    one_sided = scores().assign(label=0)
    assert summarize(one_sided, bootstrap=10) == {"v3": {"n": 600, "note": "no terminated trials resolved yet"}}
