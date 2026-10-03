"""Metrics for censored data with competing events: a trial ends terminated (cause 1), completed
(cause 2), or is still running at the snapshot (censored, 0).

Plain AUC and Brier score treat a trial that is still running as if its outcome were known. Here every
finished trial is weighted by the inverse probability of having stayed uncensored that long (IPCW), so
the heavily censored recent cohorts are scored without bias, assuming censoring is independent of the
outcome given time. Time-dependent AUC uses the competing-risk definition of Blanche, Dartigues and
Jacqmin-Gadda (2013): cases are trials terminated by t; controls are trials still running at t or
completed by t.
"""
import numpy as np


def _tally(time, event):
    """Unique times u, the number at risk just before each (time >= u), and a counter of rows at each u:
    count(mask) -> how many rows with mask True end exactly at each u."""
    time = np.asarray(time, dtype=float)
    u, idx = np.unique(time, return_inverse=True)
    at_risk = len(time) - np.searchsorted(np.sort(time), u, side="left")

    def count(mask):
        return np.bincount(idx[mask], minlength=len(u))
    return u, at_risk, count


def censoring_survival(time, event):
    """Kaplan-Meier estimate G of P(not yet censored at t). At tied times events leave before censoring.
    Returns G(t, left=False); left=True gives G(t-)."""
    event = np.asarray(event)
    u, at_risk, count = _tally(time, event)
    censored, ended = count(event == 0), count(event != 0)
    keep = censored > 0
    steps = u[keep]
    surv = np.cumprod(1 - censored[keep] / (at_risk[keep] - ended[keep]))   # those ending at u leave first

    def G(x, left: bool = False):
        x = np.asarray(x, dtype=float)
        i = np.searchsorted(steps, x, side="left" if left else "right")
        out = np.where(i == 0, 1.0, surv[np.maximum(i - 1, 0)]) if len(surv) else np.ones_like(x)
        return out.item() if out.ndim == 0 else out
    return G


def _weights(time, event, t, cause):
    """IPCW weights and the case/control indicators at horizon t."""
    time, event = np.asarray(time, dtype=float), np.asarray(event)
    G = censoring_survival(time, event)
    case = (time <= t) & (event == cause)
    control = (time > t) | ((time <= t) & (event != cause) & (event != 0))
    w = np.zeros(len(time))
    ended = (time <= t) & (event != 0)
    w[ended] = 1 / np.maximum(G(time[ended], left=True), 1e-12)
    w[time > t] = 1 / max(G(t), 1e-12)
    return case, control, w


def time_auc(time, event, risk, t: float, cause: int = 1) -> float:
    """IPCW time-dependent AUC for `cause` by t: the weighted share of (case, control) pairs ranked right."""
    case, control, w = _weights(time, event, t, cause)
    risk = np.asarray(risk, dtype=float)
    cr, cw = risk[control], w[control]
    order = np.argsort(cr)
    cr, cum = cr[order], np.concatenate([[0], np.cumsum(cw[order])])
    lo = np.searchsorted(cr, risk[case], side="left")
    hi = np.searchsorted(cr, risk[case], side="right")
    below, ties = cum[lo], cum[hi] - cum[lo]
    return float(np.sum(w[case] * (below + 0.5 * ties)) / (w[case].sum() * cw.sum()))


def brier(time, event, cif, t: float, cause: int = 1) -> float:
    """IPCW Brier score of a predicted cumulative incidence of `cause` by t (lower is better)."""
    case, _, w = _weights(time, event, t, cause)
    return float(np.mean(w * (case - np.asarray(cif, dtype=float)) ** 2))


def aalen_johansen(time, event, t: float, cause: int = 1) -> float:
    """Cumulative incidence of `cause` by t with no covariates: the competing-risk analogue of 1 - KM."""
    event = np.asarray(event)
    u, at_risk, count = _tally(time, event)
    ended, this = count(event != 0), count(event == cause)
    keep = (u <= t) & (ended > 0)
    survive = np.cumprod(1 - ended[keep] / at_risk[keep])
    before = np.concatenate([[1.0], survive[:-1]])             # still running just before each time
    return float(np.sum(before * this[keep] / at_risk[keep]))


def calibration_by_decile(time, event, cif, t: float, cause: int = 1) -> list[dict]:
    """Mean predicted cumulative incidence by decile of prediction vs the Aalen-Johansen estimate in it."""
    time, event, cif = np.asarray(time, dtype=float), np.asarray(event), np.asarray(cif, dtype=float)
    bins = np.array_split(np.argsort(cif, kind="stable"), 10)
    return [{"bin": b, "mean_predicted": round(float(cif[i].mean()), 4),
             "observed": round(aalen_johansen(time[i], event[i], t, cause), 4), "n": len(i)}
            for b, i in enumerate(bins)]
