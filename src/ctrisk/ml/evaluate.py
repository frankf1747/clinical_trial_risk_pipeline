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
    Both by Newton's method: two parameters, no regularization, no extra dependency.
    """
    y, x = np.asarray(y, dtype=float), _logit(score)
    a = 0.0
    for _ in range(iterations):                                   # intercept with offset x, slope fixed
        q = 1 / (1 + np.exp(-(x + a)))
        a -= (q - y).sum() / max((q * (1 - q)).sum(), 1e-12)
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
