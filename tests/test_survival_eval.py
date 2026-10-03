import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ctrisk.ml.survival_eval import (
    aalen_johansen,
    brier,
    calibration_by_decile,
    censoring_survival,
    time_auc,
)


def simulate(n=40000, seed=0, censor=True):
    """Two competing causes whose hazards depend on a risk score x; independent uniform censoring."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t_term = rng.exponential(1 / (0.15 * np.exp(0.8 * x)))       # higher x -> terminated sooner
    t_comp = rng.exponential(1 / 0.4, size=n)                     # completion does not depend on x
    t = np.minimum(t_term, t_comp)
    event = np.where(t_term < t_comp, 1, 2)
    if censor:
        c = rng.uniform(0, 6, size=n)
        event = np.where(c < t, 0, event)
        t = np.minimum(t, c)
    return x, t, event


def test_without_censoring_the_weights_vanish_and_auc_is_the_plain_auc():
    x, t, e = simulate(5000, censor=False)
    y = (t <= 2) & (e == 1)
    assert time_auc(t, e, x, 2.0) == pytest.approx(roc_auc_score(y, x), abs=1e-9)
    assert brier(t, e, np.full(len(t), 0.2), 2.0) == pytest.approx(np.mean((y - 0.2) ** 2))
    assert aalen_johansen(t, e, 2.0) == pytest.approx(y.mean())


def test_ipcw_recovers_the_uncensored_answers():
    x, t, e = simulate(censor=False)
    xc, tc, ec = simulate(censor=True)                            # same draws, then censored
    assert (ec == 0).mean() > 0.15
    for horizon in (1.0, 2.0, 3.0):
        truth_auc = time_auc(t, e, x, horizon)
        assert time_auc(tc, ec, xc, horizon) == pytest.approx(truth_auc, abs=0.01)
        assert aalen_johansen(tc, ec, horizon) == pytest.approx(aalen_johansen(t, e, horizon), abs=0.01)
    naive = roc_auc_score((tc <= 2) & (ec == 1), xc)              # treating censored as controls is biased
    assert abs(naive - time_auc(t, e, x, 2.0)) > abs(time_auc(tc, ec, xc, 2.0) - time_auc(t, e, x, 2.0))


def test_censoring_survival_is_a_kaplan_meier_step_function():
    t = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    e = np.array([0, 1, 0, 2, 0])                                 # censored at 1, 2 (with an event at 2), 4
    G = censoring_survival(t, e)
    assert G(0.5) == 1.0 and G(1.0) == pytest.approx(4 / 5)
    assert G(2.0, left=True) == pytest.approx(4 / 5)
    assert G(2.0) == pytest.approx(4 / 5 * (1 - 1 / 3))            # at 2 the event leaves first: 3 at risk
    assert G(5.0) == 0.0                                           # the last one at risk was censored


def test_calibration_compares_each_decile_with_its_observed_incidence():
    x, t, e = simulate(20000, censor=True)
    pred = 1 - np.exp(-0.15 * np.exp(0.8 * x) * 2)                # roughly the true CIF ordering
    rows = calibration_by_decile(t, e, pred, 2.0)
    assert len(rows) == 10 and rows[-1]["observed"] > rows[0]["observed"]
    assert all({"mean_predicted", "observed", "n"} <= set(r) for r in rows)
