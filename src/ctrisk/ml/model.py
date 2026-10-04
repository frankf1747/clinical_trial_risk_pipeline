"""Tabular inputs + text features + LightGBM as one object: fit once, save with joblib, score later.

With seeds > 1 the model is a seed ensemble: that many LightGBM fits on the same rows and features, differing
only in the random seed behind row and column subsampling, with their log-odds averaged. One fit's score
for a single trial moves noticeably from seed to seed; the average moves far less. Contributions are
averaged the same way, so they still add up to the ensemble's log-odds.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from ctrisk.ml.features import to_matrix
from ctrisk.ml.text import TextFeatures


class RiskModel:
    def __init__(self, columns: list[str], params: dict, text: TextFeatures | None, fit_text: bool = True,
                 embed=None, fit_embed: bool = True, seeds: int = 1):
        self.columns, self.params, self.text, self.fit_text = list(columns), dict(params), text, fit_text
        self.seeds = seeds
        self.embed, self.fit_embed = embed, fit_embed      # M9: EmbeddingFeatures over frame["embedding"]
        self.categories: dict | None = None
        self.lgbm: lgb.LGBMClassifier | None = None        # the first member
        self.members: list[lgb.LGBMClassifier] = []

    @property
    def _members(self) -> list[lgb.LGBMClassifier]:
        return getattr(self, "members", None) or [self.lgbm]     # models saved before the ensemble: one fit

    @property
    def feature_names(self) -> list[str]:
        embed = getattr(self, "embed", None)                 # models saved before M9 have no embed
        return self.columns + (self.text.columns if self.text else []) + (embed.columns if embed else [])

    @property
    def best_iteration(self) -> int:
        return self.lgbm.best_iteration_ or self.params["n_estimators"]

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        X, learned = to_matrix(frame, self.columns, self.categories)
        if self.categories is None:
            self.categories = learned
        if self.text:
            X = pd.concat([X, self.text.transform(frame["text"])], axis=1)
        if getattr(self, "embed", None):
            X = pd.concat([X, self.embed.transform(frame["embedding"])], axis=1)
        return X

    def fit(self, frame: pd.DataFrame, y: pd.Series, valid: tuple | None = None) -> "RiskModel":
        if self.text and self.fit_text:
            self.text.fit(frame["text"])
        if getattr(self, "embed", None) and self.fit_embed:
            self.embed.fit(frame["embedding"])
        X = self._matrix(frame)                      # learns categories from the training frame only
        kwargs = {}
        if valid is not None:
            kwargs = {"eval_X": self._matrix(valid[0]), "eval_y": valid[1],
                      "callbacks": [lgb.early_stopping(50, verbose=False)]}
        seed = self.params.get("random_state", 0)
        self.members = [lgb.LGBMClassifier(**{**self.params, "random_state": seed + i}).fit(X, y, **kwargs)
                        for i in range(self.seeds)]
        self.lgbm = self.members[0]
        return self

    def member_log_odds(self, frame: pd.DataFrame) -> np.ndarray:
        """(members, rows): each member's log-odds."""
        X = self._matrix(frame)
        return np.stack([m.predict(X, raw_score=True) for m in self._members])

    def log_odds(self, frame: pd.DataFrame) -> np.ndarray:
        return self.member_log_odds(frame).mean(axis=0)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self.log_odds(frame)))

    def contributions(self, frame: pd.DataFrame) -> np.ndarray:
        X = self._matrix(frame)
        return np.mean([m.predict(X, pred_contrib=True)[:, :-1] for m in self._members], axis=0)   # last: bias


def collapse_text(contrib: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Sum the text columns (txt_* SVD, emb_* embedding components) into one 'registration_text' column,
    appended last, for readable drivers."""
    is_text = [n.startswith(("txt_", "emb_")) for n in names]
    keep = [i for i, t in enumerate(is_text) if not t]
    text_idx = [i for i, t in enumerate(is_text) if t]
    out, out_names = contrib[:, keep], [names[i] for i in keep]
    if text_idx:
        out = np.concatenate([out, contrib[:, text_idx].sum(axis=1, keepdims=True)], axis=1)
        out_names = out_names + ["registration_text"]
    return out, out_names
