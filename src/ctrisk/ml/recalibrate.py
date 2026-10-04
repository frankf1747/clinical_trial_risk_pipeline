"""Recalibrate the survival model to the current level of termination risk.

The hazards are learned from trials that started 2008-2014. Two things are off for the trials running
since. Terminations have become more common for trials of every start year: over calendar 2020-2024,
trials that started 2015 or later were terminated 20-30% more often than the model expected, while
completions ran close to expectation. And the learned termination risks are too spread out: trials the
model rates low terminate more often than it says, and those it rates high less often.

So the per-period log-odds of terminating (vs still running) are mapped through an intercept and a slope,
and the same for completing, fit by maximum likelihood on the person-periods of a recent calendar window.
The last LAG years before the snapshot are left out of that window: sponsors record that a trial ended
months after it did, so in calendar 2025-2026 both terminations and completions fall well short of
expectation (completions too, which is reporting lag rather than lower risk).

The method is checked the way the lookup uses it. At each landmark date, trials then running get their
risk of termination in the next 2 years, recalibrated only on calendar time ending LAG years before the
landmark, as at the snapshot. That is compared with what happened to them (Aalen-Johansen).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ctrisk.ml.survival import (
    IDENTITY,
    SNAPSHOT,
    YEAR,
    SurvivalModel,
    expand,
    recalibrated,
)
from ctrisk.ml.survival_eval import aalen_johansen, calibration_by_decile, time_auc

LAG = 1.7            # years before the snapshot whose endings are not yet fully recorded
WINDOW = 3.0         # years of calendar time the recalibration is fit on
HORIZON = 2.0        # the lookup's window: termination in the next 2 years
MAX_ELAPSED = 6.0    # as the lookup: a 2-year window must end inside the model's 8 years
LANDMARKS = ("2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01")


def _years(x: float) -> pd.Timedelta:
    return pd.Timedelta(days=x * YEAR)


def person_periods(start, time, event, width: float, periods: int):
    """Person-period rows (trial, period, outcome) and the calendar date at the middle of each period."""
    rows, period, y = expand(time, event, width, periods)
    mid = pd.DatetimeIndex(start)[rows] + pd.to_timedelta((period + 0.5) * width * YEAR, unit="D")
    return rows, period, y, mid


def fit_recalibration(h: np.ndarray, y: np.ndarray) -> tuple:
    """Intercept and slope for terminating and for completing, by maximum likelihood on these person-period
    rows. h: (rows, 3) hazards as learned; y: outcome of each row (0 running, 1 terminated, 2 completed)."""
    y = np.asarray(y)
    if len(y) == 0 or not ((y == 1).any() and (y == 2).any()):
        return IDENTITY
    pick = np.arange(len(y))

    def nll(p):
        return -np.log(np.maximum(recalibrated(h, p)[pick, y], 1e-12)).sum()
    fit = minimize(nll, np.array(IDENTITY), method="L-BFGS-B")
    return tuple(float(v) for v in fit.x)


def window_fit(h, rows, period, y, mid, end: pd.Timestamp, years: float = WINDOW) -> tuple[tuple, int]:
    """Recalibration fit on the periods whose midpoint falls in [end - years, end), and how many there are."""
    m = np.asarray((mid >= end - _years(years)) & (mid < end))
    return fit_recalibration(h[rows[m], period[m]], y[m]), int(m.sum())


def at_landmark(start, time, event, date: pd.Timestamp, max_elapsed: float = MAX_ELAPSED):
    """Trials running at `date` (started before it, not yet ended, at most max_elapsed years in): a mask,
    their elapsed years, and their time and event counted from the date."""
    elapsed = np.asarray((date - pd.DatetimeIndex(start)).days, dtype=float) / YEAR
    time, event = np.asarray(time, dtype=float), np.asarray(event)
    keep = (elapsed > 0) & (time > elapsed) & (elapsed <= max_elapsed)
    return keep, elapsed[keep], time[keep] - elapsed[keep], event[keep]


def _rounded(params) -> list[float]:
    return [round(float(v), 3) for v in params]


def backtest(model: SurvivalModel, h, start, time, event, yes_no=None, landmarks=LANDMARKS, lag: float = LAG,
             horizon: float = HORIZON) -> list[dict]:
    """At each landmark: next-`horizon`-year risk of the trials then running, as learned and recalibrated
    on calendar time ending `lag` years before the landmark, against the observed incidence. `yes_no`:
    another score per trial (the yes/no model's), ranked the same way for comparison."""
    rows, period, y, mid = person_periods(start, time, event, model.width, model.periods)
    out = []
    for date in map(pd.Timestamp, landmarks):
        end = date - _years(lag)
        params, n_periods = window_fit(h, rows, period, y, mid, end)
        keep, elapsed, rest, ev = at_landmark(start, time, event, date)
        if keep.sum() == 0:
            continue
        before = model.conditional_cif(None, elapsed, horizon, h=h[keep])
        after = model.conditional_cif(None, elapsed, horizon, h=recalibrated(h[keep], params))
        row = {"landmark": str(date.date()), "fit_window": [str((end - _years(WINDOW)).date()), str(end.date())],
               "fit_periods": n_periods, "params": _rounded(params), "running": int(keep.sum()),
               "observed": round(aalen_johansen(rest, ev, horizon), 4),
               "predicted_as_learned": round(float(before.mean()), 4),
               "predicted_recalibrated": round(float(after.mean()), 4),
               "time_auc": round(time_auc(rest, ev, after, horizon), 4),
               "calibration": calibration_by_decile(rest, ev, after, horizon),
               "calibration_as_learned": calibration_by_decile(rest, ev, before, horizon)}
        if yes_no is not None:
            row["time_auc_yes_no_model"] = round(time_auc(rest, ev, np.asarray(yes_no)[keep], horizon), 4)
        out.append(row)
    return out


def observed_vs_expected(h, start, time, event, width: float, periods: int) -> list[dict]:
    """Terminations and completions observed vs expected (as learned) by calendar year."""
    rows, period, y, mid = person_periods(start, time, event, width, periods)
    d = pd.DataFrame({"year": mid.year, "n": 1, "term": y == 1, "comp": y == 2,
                      "exp_term": h[rows, period, 1], "exp_comp": h[rows, period, 2]})
    g = d.groupby("year").sum()
    return [{"year": int(yr), "periods_at_risk": int(r.n),
             "terminated_observed_over_expected": round(float(r.term / r.exp_term), 3),
             "completed_observed_over_expected": round(float(r.comp / r.exp_comp), 3)} for yr, r in g.iterrows()]


def recalibrate(model: SurvivalModel, later: pd.DataFrame, yes_no=None, snapshot=SNAPSHOT, lag: float = LAG) -> dict:
    """Backtest the recalibration on `later` (trials not used to fit the model, with start, time, event),
    then set model.recalibration from the calendar window ending `lag` years before the snapshot."""
    model.recalibration = IDENTITY
    h = model.hazards(later, raw=True)
    start, time, event = later["start"].to_numpy(), later["time"].to_numpy(), later["event"].to_numpy()
    report = {"lag_years": lag, "window_years": WINDOW, "horizon_years": HORIZON,
              "by_calendar_year": observed_vs_expected(h, start, time, event, model.width, model.periods),
              "backtest": backtest(model, h, start, time, event, yes_no=yes_no, lag=lag)}
    rows, period, y, mid = person_periods(start, time, event, model.width, model.periods)
    end = pd.Timestamp(snapshot) - _years(lag)
    params, n_periods = window_fit(h, rows, period, y, mid, end)
    model.recalibration = params
    report["serving"] = {"fit_window": [str((end - _years(WINDOW)).date()), str(end.date())],
                         "fit_periods": n_periods, "params": _rounded(params)}
    return report
