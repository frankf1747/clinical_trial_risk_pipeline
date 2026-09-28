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
