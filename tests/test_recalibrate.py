import numpy as np
import pandas as pd
import pytest

from ctrisk.ml import recalibrate as rc
from ctrisk.ml.survival import IDENTITY, SNAPSHOT, SurvivalModel, recalibrated

TRUE = (0.4, 0.7, -0.1, 1.0)              # the world vs the learned hazards: more terminations, less spread


def learned_hazards(n, periods=16, seed=0):
    rng = np.random.default_rng(seed)
    term = np.clip(rng.lognormal(np.log(0.03), 0.8, size=n), 0.002, 0.3)
    comp = np.clip(rng.lognormal(np.log(0.12), 0.5, size=n), 0.02, 0.4)
    h = np.stack([1 - term - comp, term, comp], axis=-1)
    return np.repeat(h[:, None, :], periods, axis=1)                      # same each period, per trial


def simulate(h_true, start, seed=1, width=0.5):
    """Time and event per trial from per-period hazards, censored at the snapshot."""
    rng = np.random.default_rng(seed)
    n, periods, _ = h_true.shape
    follow = (pd.Timestamp(SNAPSHOT) - pd.DatetimeIndex(start)).days.to_numpy() / 365.25
    time, event = np.minimum(follow, periods * width), np.zeros(n, dtype=int)
    for k in range(periods):
        open_ = (event == 0) & ((k + 1) * width <= follow)
        u = rng.random(n)
        p = h_true[:, k]
        ended = open_ & (u > p[:, 0])
        event[ended] = np.where(u[ended] < p[ended, 0] + p[ended, 1], 1, 2)
        time[ended] = (k + 0.5) * width
    return time, event


class Fixed(SurvivalModel):
    """A survival model whose learned hazards are given."""
    def __init__(self, h):
        super().__init__(["x"], {}, text=None)
        self.h = h

    def hazards(self, frame, chunk=4000, raw=False):
        return self.h if raw else recalibrated(self.h, self.recalibration)


@pytest.fixture(scope="module")
def world():
    n = 20000
    rng = np.random.default_rng(2)
    start = pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.uniform(0, 9.5 * 365.25, n), unit="D")
    h = learned_hazards(n)
    time, event = simulate(recalibrated(h, TRUE), start)
    return h, pd.DataFrame({"start": start, "time": time, "event": event})


def test_identity_leaves_hazards_alone_and_any_map_gives_probabilities():
    h = learned_hazards(50)
    assert recalibrated(h, IDENTITY) is h
    r = recalibrated(h, (0.5, 0.6, -0.2, 1.1))
    assert np.allclose(r.sum(axis=-1), 1) and (r > 0).all()
    z = np.log(h[:, 0, 1] / h[:, 0, 0])
    zr = np.log(r[:, 0, 1] / r[:, 0, 0])
    assert np.allclose(zr, 0.5 + 0.6 * z)                                 # termination log-odds: a + b * z


def test_maximum_likelihood_recovers_the_distortion(world):
    h, w = world
    rows, period, y, _ = rc.person_periods(w["start"], w["time"], w["event"], 0.5, 16)
    fit = rc.fit_recalibration(h[rows, period], y)
    assert fit == pytest.approx(TRUE, abs=0.08)
    assert rc.fit_recalibration(h[:0, 0], y[:0]) == IDENTITY              # nothing to fit on


def test_at_landmark_keeps_trials_running_then_and_counts_time_from_it():
    start = pd.to_datetime(["2018-01-01", "2018-01-01", "2020-06-01", "2010-01-01"])
    time, event = np.array([1.0, 3.0, 1.0, 12.0]), np.array([1, 2, 0, 0])
    keep, elapsed, rest, ev = rc.at_landmark(start, time, event, pd.Timestamp("2020-01-01"))
    assert keep.tolist() == [False, True, False, False]   # ended before; not started; running 10 years (> 6)
    assert elapsed[0] == pytest.approx(2.0, abs=0.01) and rest[0] == pytest.approx(1.0, abs=0.01) and ev[0] == 2


def test_backtest_moves_predictions_to_what_happened(world):
    h, w = world
    model = Fixed(h)
    out = rc.backtest(model, h, w["start"], w["time"], w["event"], landmarks=("2020-01-01", "2021-01-01"))
    for b in out:
        assert abs(b["predicted_recalibrated"] - b["observed"]) < abs(b["predicted_as_learned"] - b["observed"]) / 2
        assert b["fit_window"][1] < b["landmark"] and len(b["calibration"]) == 10


def test_recalibrate_sets_the_model_from_the_window_before_the_reporting_lag(world):
    h, w = world
    model = Fixed(h)
    report = rc.recalibrate(model, w)
    assert report["serving"]["fit_window"][1] == str((pd.Timestamp(SNAPSHOT) - pd.Timedelta(days=rc.LAG * 365.25)).date())
    assert model.recalibration == pytest.approx(TRUE, abs=0.1)
    assert not np.allclose(model.hazards(None), h)                       # served hazards carry it
    years = {r["year"]: r for r in report["by_calendar_year"]}
    assert years[2020]["terminated_observed_over_expected"] > 1.2
