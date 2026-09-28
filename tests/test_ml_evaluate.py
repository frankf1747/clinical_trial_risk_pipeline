import numpy as np

from ctrisk.ml.evaluate import bootstrap_auc_ci, calibration, metrics, precision_at_top

Y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
PERFECT = np.array([0.9, 0.1, 0.8, 0.2, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1])


def test_precision_at_top_counts_terminations_among_highest_scores():
    assert precision_at_top(Y, PERFECT, frac=0.2) == 1.0     # top 2 are both terminated
    assert precision_at_top(Y, -PERFECT, frac=0.2) == 0.0


def test_metrics_on_a_perfect_ranking():
    m = metrics(Y, PERFECT)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert (m["n"], m["base_rate"]) == (10, 0.2)


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    score = (y * 0.5 + rng.random(500)) / 1.5
    lo, hi = bootstrap_auc_ci(y, score, n=200, seed=0)
    assert (lo, hi) == bootstrap_auc_ci(y, score, n=200, seed=0)
    assert lo < metrics(y, score)["roc_auc"] < hi


def test_calibration_bins_cover_every_row():
    rows = calibration(Y, PERFECT, bins=2)
    assert sum(r["n"] for r in rows) == len(Y)


def test_calibration_ignores_pandas_index_alignment():
    import pandas as pd
    y = pd.Series(Y, index=range(500, 510))      # a slice of a larger frame
    rows = calibration(y, pd.Series(PERFECT), bins=2)
    assert sum(r["observed"] * r["n"] for r in rows) == Y.sum()
