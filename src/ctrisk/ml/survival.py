"""When and how a trial ends: a discrete-time competing-risks model.

A trial ends terminated (event 1), ends completed (event 2), or is still running at the AACT snapshot
(censored, 0). Each trial becomes one row per 6-month period it was at risk; a LightGBM multiclass model
predicts, for a trial in a given period, P(still running), P(terminated), P(completed) in that period,
from the same features as the yes/no model plus the period index. From those hazards:

    S(k)    = prod_{j<=k} P(still running in j)                       probability of still running after k
    CIF(t)  = sum_{k < t/width} S(k-1) * P(terminated in k)           probability of termination by t
    given a trial has already run m periods, its risk over the next w: sum_{k=m}^{m+w-1} S(k-1) h(k) / S(m-1)

Running trials count as at risk for the periods they were observed, so they inform the model instead of
being dropped, and nothing about the end date enters the features.

`recalibration` maps the log-odds of terminating and of completing (vs still running) in every period
through an intercept and a slope each; it is fit afterwards on recent calendar time (ctrisk.ml.recalibrate)
so the probabilities reflect today's level of risk. IDENTITY leaves the hazards as learned.

With seeds > 1 the hazards come from a seed ensemble: that many fits differing only in the random seed, with
their per-period class log-odds averaged before the softmax. One fit's next-2-year risk for a single trial
moves a lot from seed to seed; the average far less.
"""
import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd

from ctrisk.ml.features import to_matrix
from ctrisk.ml.text import TextFeatures

SNAPSHOT = dt.date(2026, 9, 26)          # AACT snapshot the labels come from: running trials are censored here
WIDTH = 0.5                              # years per period
PERIODS = 16                             # 8 years; events later than that are treated as censored at 8 years
YEAR = 365.25


def durations(frame: pd.DataFrame) -> pd.DataFrame:
    """time (years from start) and event (1 terminated, 2 completed, 0 running) per trial, same index.
    Finished trials whose record has no completion date are dropped: their duration is unknown."""
    start = pd.to_datetime(frame["start_date"])
    end = pd.to_datetime(frame["end_date"])
    finished = frame["label"].notna()
    keep = ~finished | end.notna()
    snapshot = pd.Timestamp(SNAPSHOT)
    time = np.where(finished, (end - start).dt.days, (snapshot - start).dt.days) / YEAR
    event = np.where(~finished, 0, np.where(frame["label"] == 1, 1, 2))
    out = pd.DataFrame({"time": np.clip(time, 0, None), "event": event}, index=frame.index)
    return out[keep]


IDENTITY = (0.0, 1.0, 0.0, 1.0)         # intercept, slope for terminating; intercept, slope for completing


def recalibrated(h: np.ndarray, params=IDENTITY) -> np.ndarray:
    """Hazards (..., 3) with log(P(terminated) / P(running)) -> a1 + b1 * it, and the same for completed."""
    a1, b1, a2, b2 = params
    if np.array_equal(params, IDENTITY):
        return h
    log_run = np.log(np.maximum(h[..., 0], 1e-12))
    z1 = np.log(np.maximum(h[..., 1], 1e-12)) - log_run
    z2 = np.log(np.maximum(h[..., 2], 1e-12)) - log_run
    logits = np.stack([np.zeros_like(z1), a1 + b1 * z1, a2 + b2 * z2], axis=-1)
    w = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return w / w.sum(axis=-1, keepdims=True)


def expand(time: np.ndarray, event: np.ndarray, width: float = WIDTH, periods: int = PERIODS):
    """Person-period rows: (trial position, period, outcome in that period: 0 none, 1 terminated, 2 completed).
    A censored trial contributes only the periods it was observed in full."""
    k = np.floor(np.asarray(time, dtype=float) / width).astype(int)
    event = np.asarray(event).copy()
    event[k >= periods] = 0                               # ended after the horizon: censored at the horizon
    k = np.minimum(k, periods)
    n_rows = np.where(event > 0, k + 1, k)                # the period it ended in counts as at risk
    rows = np.repeat(np.arange(len(k)), n_rows)
    period = np.arange(n_rows.sum()) - np.repeat(np.cumsum(n_rows) - n_rows, n_rows)
    y = np.zeros(len(rows), dtype=int)
    last = np.cumsum(n_rows) - 1
    ended = event > 0
    y[last[ended]] = event[ended]
    return rows, period, y


class SurvivalModel:
    def __init__(self, columns: list[str], params: dict, text: TextFeatures | None,
                 width: float = WIDTH, periods: int = PERIODS, seeds: int = 1):
        self.columns, self.params, self.text = list(columns), dict(params), text
        self.width, self.periods, self.seeds = width, periods, seeds
        self.categories: dict | None = None
        self.lgbm: lgb.LGBMClassifier | None = None        # the first member
        self.members: list[lgb.LGBMClassifier] = []
        self.recalibration = IDENTITY

    def _base(self, frame: pd.DataFrame) -> pd.DataFrame:
        X, learned = to_matrix(frame.reset_index(drop=True), self.columns, self.categories)
        if self.categories is None:
            self.categories = learned
        if self.text:
            X = pd.concat([X, self.text.transform(frame["text"].reset_index(drop=True))], axis=1)
        return X

    def _periods(self, X: pd.DataFrame, rows: np.ndarray, period: np.ndarray) -> pd.DataFrame:
        out = X.iloc[rows].reset_index(drop=True)
        out["period"] = period
        return out

    def fit(self, frame: pd.DataFrame, time, event, valid: tuple | None = None) -> "SurvivalModel":
        if self.text:
            self.text.fit(frame["text"])
        X = self._base(frame)
        rows, period, y = expand(time, event, self.width, self.periods)
        kwargs = {}
        if valid is not None:
            vframe, vtime, vevent = valid
            vr, vp, vy = expand(vtime, vevent, self.width, self.periods)
            kwargs = {"eval_X": self._periods(self._base(vframe), vr, vp), "eval_y": vy,
                      "callbacks": [lgb.early_stopping(50, verbose=False)]}
        Xp, seed = self._periods(X, rows, period), self.params.get("random_state", 0)
        self.members = [lgb.LGBMClassifier(objective="multiclass", num_class=3,
                                           **{**self.params, "random_state": seed + i}).fit(Xp, y, **kwargs)
                        for i in range(self.seeds)]
        self.lgbm = self.members[0]
        return self

    @property
    def best_iteration(self) -> int:
        return self.lgbm.best_iteration_ or self.params["n_estimators"]

    def _predict(self, Xp: pd.DataFrame, members: slice | None) -> np.ndarray:
        """Class probabilities, from the members' log-odds averaged."""
        fitted = getattr(self, "members", None) or [self.lgbm]             # models saved before the ensemble
        logits = np.mean([m.predict(Xp, raw_score=True) for m in fitted[members or slice(None)]], axis=0)
        w = np.exp(logits - logits.max(axis=1, keepdims=True))
        return w / w.sum(axis=1, keepdims=True)

    def hazards(self, frame: pd.DataFrame, chunk: int = 4000, raw: bool = False,
                members: slice | None = None) -> np.ndarray:
        """(trials, periods, 3): P(still running, terminated, completed) in each period, if at risk in it.
        raw=True: as learned, without the recalibration. members: a slice of the ensemble (default all)."""
        X = self._base(frame)
        out = np.empty((len(X), self.periods, 3))
        for lo in range(0, len(X), chunk):
            idx = np.arange(lo, min(lo + chunk, len(X)))
            rows, period = np.repeat(idx, self.periods), np.tile(np.arange(self.periods), len(idx))
            out[idx] = self._predict(self._periods(X, rows, period), members).reshape(len(idx), self.periods, 3)
        return out if raw else recalibrated(out, getattr(self, "recalibration", IDENTITY))   # older pickles: none

    @staticmethod
    def _incidence(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per period: probability of still running just before it, and of terminating in it."""
        still = np.cumprod(h[:, :, 0], axis=1)
        before = np.concatenate([np.ones((len(h), 1)), still[:, :-1]], axis=1)
        return before, before * h[:, :, 1]

    def cif(self, frame: pd.DataFrame, years: list[float], h: np.ndarray | None = None) -> np.ndarray:
        """(trials, len(years)): probability of termination within each horizon of the start."""
        _, inc = self._incidence(self.hazards(frame) if h is None else h)
        cum = np.cumsum(inc, axis=1)
        ends = [min(round(y / self.width), self.periods) - 1 for y in years]
        return cum[:, ends]

    def conditional_cif(self, frame: pd.DataFrame, elapsed: np.ndarray, window: float,
                        h: np.ndarray | None = None) -> np.ndarray:
        """Probability of termination within `window` years from now, for trials already running `elapsed`
        years. Windows reaching past the horizon are cut at it."""
        before, inc = self._incidence(self.hazards(frame) if h is None else h)
        m = np.clip(np.floor(np.asarray(elapsed, dtype=float) / self.width).astype(int), 0, self.periods - 1)
        w = round(window / self.width)
        cum = np.concatenate([np.zeros((len(inc), 1)), np.cumsum(inc, axis=1)], axis=1)
        stop = np.minimum(m + w, self.periods)
        rows = np.arange(len(inc))
        return (cum[rows, stop] - cum[rows, m]) / np.maximum(before[rows, m], 1e-12)
