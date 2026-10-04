import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ctrisk.ml.survival import SNAPSHOT, SurvivalModel, durations, expand
from ctrisk.ml.survival_eval import time_auc


def test_durations_censor_running_trials_at_the_snapshot_and_drop_finished_ones_without_an_end():
    f = pd.DataFrame({"nct_id": ["T", "C", "A", "F", "X"],
                      "start_date": pd.to_datetime(["2010-01-01", "2010-01-01", "2025-09-26", "2030-01-01", "2010-01-01"]),
                      "end_date": pd.to_datetime(["2011-07-02", "2012-01-01", None, None, None]),
                      "label": [1, 0, np.nan, np.nan, 1]})
    d = durations(f)
    assert list(d.index) == [0, 1, 2, 3]                                 # X: finished, no end date
    assert d["event"].tolist() == [1, 2, 0, 0]
    assert d.loc[0, "time"] == pytest.approx(1.5, abs=0.01)
    assert d.loc[2, "time"] == pytest.approx((SNAPSHOT - dt.date(2025, 9, 26)).days / 365.25)
    assert d.loc[3, "time"] == 0.0                                        # anticipated start: no time at risk yet


def test_expand_gives_one_row_per_period_at_risk():
    rows, period, y = expand(np.array([1.2, 1.2, 9.0, 0.3]), np.array([1, 0, 2, 0]), width=0.5, periods=16)
    by = pd.DataFrame({"r": rows, "p": period, "y": y}).groupby("r")
    assert by.get_group(0)[["p", "y"]].values.tolist() == [[0, 0], [1, 0], [2, 1]]   # ended in its 3rd period
    assert by.get_group(1)["p"].tolist() == [0, 1]                       # censored mid-period 3: 2 full periods
    assert by.get_group(2)["y"].eq(0).all() and len(by.get_group(2)) == 16   # past the horizon: censored there
    assert 3 not in by.groups                                             # censored inside its first period


def synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t_term = rng.exponential(1 / (0.2 * np.exp(0.9 * x)))
    t_comp = rng.exponential(1 / 0.5, size=n)
    c = rng.uniform(0, 6, size=n)
    t = np.minimum.reduce([t_term, t_comp, c])
    event = np.select([t == c, t_term < t_comp], [0, 1], 2)
    frame = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "phase": rng.choice(["PHASE1", "PHASE2"], n),
                          "text": np.where(x > 1, "slow accrual", "large trial")})
    return frame, t, event


@pytest.fixture(scope="module")
def fitted():
    frame, t, e = synthetic()
    params = {"n_estimators": 80, "num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 50, "verbose": -1}
    return SurvivalModel(["x", "noise", "phase"], params, text=None).fit(frame, t, e), synthetic(seed=1)


def test_cumulative_incidence_ranks_trials_and_grows_with_time(fitted):
    model, (frame, t, e) = fitted
    cif = model.cif(frame, [1, 2, 3])
    assert cif.shape == (len(frame), 3) and ((0 <= cif) & (cif <= 1)).all()
    assert (np.diff(cif, axis=1) >= -1e-12).all()                         # never decreases with time
    assert time_auc(t, e, cif[:, 1], 2.0) > 0.7                          # x drives termination


def test_conditional_risk_for_a_running_trial_is_a_proper_probability(fitted):
    model, (frame, _, _) = fitted
    later = model.conditional_cif(frame, elapsed=np.full(len(frame), 2.0), window=2.0)
    fresh = model.conditional_cif(frame, elapsed=np.zeros(len(frame)), window=2.0)
    assert ((0 <= later) & (later <= 1)).all()
    assert np.allclose(fresh, model.cif(frame, [2])[:, 0])               # from the start, it is the plain CIF


def test_a_seed_ensemble_averages_class_log_odds_and_one_member_matches_a_single_fit():
    frame, t, e = synthetic(2000)
    params = {"n_estimators": 40, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 50, "verbose": -1,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": 0}
    one = SurvivalModel(["x", "noise", "phase"], params, text=None).fit(frame, t, e)
    ens = SurvivalModel(["x", "noise", "phase"], params, text=None, seeds=3).fit(frame, t, e)
    head = frame.head(20)
    assert len(ens.members) == 3 and np.allclose(ens.hazards(head, members=slice(0, 1)), one.hazards(head))
    h = ens.hazards(head)
    assert np.allclose(h.sum(axis=-1), 1) and not np.allclose(h, one.hazards(head))
    del one.members                                                      # as a pickle saved before the ensemble
    assert np.allclose(one.hazards(head), ens.hazards(head, members=slice(0, 1)))
