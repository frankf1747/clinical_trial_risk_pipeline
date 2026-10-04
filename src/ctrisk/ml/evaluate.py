"""Ranking and calibration metrics for a termination-risk score."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def precision_at_top(y, score, frac: float = 0.10) -> float:
    """Share of terminated trials among the highest-scored `frac` of trials."""
    k = max(1, round(len(score) * frac))
    return float(np.asarray(y)[np.argsort(-np.asarray(score))[:k]].mean())


def metrics(y, score) -> dict:
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    return {"n": len(y),
            "positives": int(y.sum()),
            "base_rate": round(float(y.mean()), 4),
            "roc_auc": round(float(roc_auc_score(y, score)), 4),
            "pr_auc": round(float(average_precision_score(y, score)), 4),
            "precision_top_10pct": round(precision_at_top(y, score), 4),
            "brier": round(float(brier_score_loss(y, score)), 4)}


def bootstrap_auc_ci(y, score, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval for ROC AUC from resampling the evaluation set."""
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], score[i]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def calibration(y, score, bins: int = 10) -> list[dict]:
    """Mean predicted risk vs observed termination rate, by score quantile."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(score, dtype=float)})  # no index alignment
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    grouped = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return [{"bin": int(b), "mean_predicted": round(r.mean_predicted, 4),
             "observed": round(r.observed, 4), "n": int(r.n)} for b, r in grouped.iterrows()]


def _logit(score) -> np.ndarray:
    p = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_fit(y, score, iterations: int = 50) -> dict:
    """Calibration intercept and slope on the logit scale (Steyerberg; Van Calster et al. 2019).

    intercept: calibration-in-the-large, the shift that makes mean predicted risk match the observed
               rate with the slope held at 1. 0 is ideal; > 0 means risk is under-predicted overall.
    slope:     from regressing the outcome on logit(score). 1 is ideal; < 1 means the scores are more
               extreme than the outcomes justify (too confident), > 1 too timid.
    Intercept by bisection, slope by damped Newton: no regularization, no extra dependency.
    """
    y, x = np.asarray(y, dtype=float), _logit(score)
    lo, hi = -30.0, 30.0                 # intercept with offset x, slope fixed: mean of sigmoid(x + a) rises with a,
    for _ in range(100):                 # so bisection always converges (Newton overshot on near-certain scores)
        a = (lo + hi) / 2
        if (1 / (1 + np.exp(-np.clip(x + a, -700, 700)))).mean() < y.mean():
            lo = a
        else:
            hi = a
    X = np.column_stack([np.ones_like(x), x])

    def loglik(b):                                                # stable: log(1 + e^z) via logaddexp
        z = X @ b
        return float((y * z - np.logaddexp(0, z)).sum())

    beta = np.array([0.0, 1.0])
    for _ in range(iterations):                                   # logistic regression y ~ 1 + x
        q = 1 / (1 + np.exp(-np.clip(X @ beta, -700, 700)))
        hessian = X.T @ (X * (q * (1 - q))[:, None])
        try:
            step = np.linalg.solve(hessian, X.T @ (y - q))
        except np.linalg.LinAlgError:                             # (near-)separated: keep the last estimate
            break
        current, t = loglik(beta), 1.0
        while t > 1e-8 and not loglik(beta + t * step) >= current:   # step-halving keeps Newton from overshooting
            t /= 2
        if t <= 1e-8:
            break
        beta = beta + t * step
        if np.abs(t * step).max() < 1e-10:
            break
    return {"intercept": round(float(a), 4), "slope": round(float(beta[1]), 4)}


def _percentile(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(reference)
    return 100 * np.searchsorted(ref, scores, side="left") / len(ref)


def refit_agreement(a: np.ndarray, b: np.ndarray, reference: np.ndarray) -> dict:
    """How far two independent fits' log-odds for the same trials disagree: rank correlation, change in
    probability, and change in percentile among the reference rows (active trials, as the lookup ranks)."""
    pa, pb = 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b))
    dp = np.abs(pa - pb)
    dpct = np.abs(_percentile(a, a[reference]) - _percentile(b, b[reference]))
    return {"spearman": round(float(pd.Series(a).corr(pd.Series(b), method="spearman")), 4),
            "median_abs_change": round(float(np.median(dp)), 4),
            "p95_abs_change": round(float(np.quantile(dp, 0.95)), 4),
            "median_percentile_change": round(float(np.median(dpct)), 2),
            "share_moved_over_5_points": round(float(np.mean(dpct > 5)), 4),
            "share_moved_over_10_points": round(float(np.mean(dpct > 10)), 4)}


def subgroup_auc(y: pd.Series, score: pd.Series, groups: pd.Series, min_n: int = 100) -> dict:
    """ROC AUC per group, for groups with at least min_n rows and both outcomes. Inputs share an index."""
    out = {}
    labels = groups.astype(object).where(groups.notna(), "missing").astype(str)
    for key in sorted(labels.unique()):
        idx = labels.index[labels == key]
        if len(idx) >= min_n and y[idx].nunique() > 1:
            out[key] = {"n": len(idx), "positives": int(y[idx].sum()),
                        "roc_auc": round(float(roc_auc_score(y[idx], score[idx])), 4)}
    return out
