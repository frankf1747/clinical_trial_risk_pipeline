import numpy as np
import pytest

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


def calibrated(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.6, n)
    return (rng.random(n) < p).astype(int), p


def test_metrics_count_positives():
    assert metrics(Y, PERFECT)["positives"] == 2


def test_calibration_fit_is_identity_for_a_calibrated_score():
    from ctrisk.ml.evaluate import calibration_fit
    y, p = calibrated()
    fit = calibration_fit(y, p)
    assert abs(fit["slope"] - 1) < 0.05 and abs(fit["intercept"]) < 0.05


def test_calibration_fit_flags_an_overconfident_score():
    from ctrisk.ml.evaluate import calibration_fit
    y, p = calibrated()
    too_sure = 1 / (1 + np.exp(-2 * np.log(p / (1 - p))))   # same ranking, twice the confidence
    assert 0.4 < calibration_fit(y, too_sure)["slope"] < 0.6


def test_calibration_fit_sees_under_prediction_in_the_intercept():
    from ctrisk.ml.evaluate import calibration_fit
    y, p = calibrated()
    assert calibration_fit(y, p / 2)["intercept"] > 0.5      # predicted risk too low overall


def test_subgroup_auc_skips_small_or_single_class_groups():
    import pandas as pd

    from ctrisk.ml.evaluate import subgroup_auc
    y = pd.Series([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], index=range(100, 110))
    p = pd.Series(PERFECT, index=y.index)
    groups = pd.Series(["big"] * 8 + [None] * 2, index=y.index)
    assert subgroup_auc(y, p, groups, min_n=5) == {"big": {"n": 8, "positives": 2, "roc_auc": 1.0}}
    assert subgroup_auc(y, p, groups, min_n=2)["big"]["n"] == 8     # 'missing' has one class: skipped


def test_calibration_fit_converges_on_a_small_fold_with_few_positives():
    from ctrisk.ml.evaluate import calibration_fit
    # (score, trials, terminated): a synthetic test fold where plain Newton from slope 1 overshot to a singular Hessian
    rows = [(6e-5, 173, 0), (3e-4, 2, 0), (6e-4, 9, 0), (9e-4, 17, 0), (3.4e-3, 9, 0), (4.2e-3, 5, 1), (0.02, 8, 2),
            (0.12, 1, 0), (0.3, 1, 0), (0.35, 1, 0), (0.375, 2, 1), (0.49, 1, 1), (0.55, 1, 0), (0.73, 1, 0), (0.8, 1, 1)]
    p = np.repeat([r[0] for r in rows], [r[1] for r in rows])
    y = np.concatenate([[1] * k + [0] * (n - k) for _, n, k in rows])
    assert abs(calibration_fit(y, p)["slope"] - 0.5882) < 0.001          # matches unpenalized sklearn


def test_calibration_intercept_is_found_even_when_newton_would_overshoot():
    from ctrisk.ml.evaluate import calibration_fit
    p = np.full(1000, 0.999)                                    # far too confident: 10% observed vs 99.9%
    y = (np.arange(1000) < 100).astype(float)
    expected = np.log(0.1 / 0.9) - np.log(0.999 / 0.001)        # shift that makes the mean match
    assert calibration_fit(y, p)["intercept"] == pytest.approx(expected, abs=1e-3)
