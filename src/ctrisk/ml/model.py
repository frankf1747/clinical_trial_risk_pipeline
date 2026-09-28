"""Tabular inputs + text features + LightGBM as one object: fit once, save with joblib, score later."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from ctrisk.ml.features import to_matrix
from ctrisk.ml.text import TextFeatures


class RiskModel:
    def __init__(self, columns: list[str], params: dict, text: TextFeatures | None, fit_text: bool = True):
        self.columns, self.params, self.text, self.fit_text = list(columns), dict(params), text, fit_text
        self.categories: dict | None = None
        self.lgbm: lgb.LGBMClassifier | None = None

    @property
    def feature_names(self) -> list[str]:
        return self.columns + (self.text.columns if self.text else [])

    @property
    def best_iteration(self) -> int:
        return self.lgbm.best_iteration_ or self.params["n_estimators"]

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        X, learned = to_matrix(frame, self.columns, self.categories)
        if self.categories is None:
            self.categories = learned
        if self.text:
            X = pd.concat([X, self.text.transform(frame["text"])], axis=1)
        return X

    def fit(self, frame: pd.DataFrame, y: pd.Series, valid: tuple | None = None) -> "RiskModel":
        if self.text and self.fit_text:
            self.text.fit(frame["text"])
        X = self._matrix(frame)                      # learns categories from the training frame only
        kwargs = {}
        if valid is not None:
            kwargs = {"eval_X": self._matrix(valid[0]), "eval_y": valid[1],
                      "callbacks": [lgb.early_stopping(50, verbose=False)]}
        self.lgbm = lgb.LGBMClassifier(**self.params).fit(X, y, **kwargs)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.lgbm.predict_proba(self._matrix(frame))[:, 1]

    def contributions(self, frame: pd.DataFrame) -> np.ndarray:
        return self.lgbm.predict(self._matrix(frame), pred_contrib=True)[:, :-1]   # last column is bias


def collapse_text(contrib: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Sum the txt_* SVD columns into one 'registration_text' column, appended last, for readable drivers."""
    is_text = [n.startswith("txt_") for n in names]
    keep = [i for i, t in enumerate(is_text) if not t]
    text_idx = [i for i, t in enumerate(is_text) if t]
    out, out_names = contrib[:, keep], [names[i] for i in keep]
    if text_idx:
        out = np.concatenate([out, contrib[:, text_idx].sum(axis=1, keepdims=True)], axis=1)
        out_names = out_names + ["registration_text"]
    return out, out_names
