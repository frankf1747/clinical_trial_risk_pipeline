import numpy as np
import pandas as pd

from ctrisk.ml.model import RiskModel, collapse_text
from ctrisk.ml.text import TextFeatures

PARAMS = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 10, "verbose": -1,
          "random_state": 0}


def frame(n, seed):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    return pd.DataFrame({
        "phase": rng.choice(["PHASE1", "PHASE2"], n),
        "n_countries": signal,
        "text": [("risky " if s > 0.5 else "calm ") + rng.choice(["aa", "bb", "cc"]) for s in signal],
    }), pd.Series((signal + rng.normal(scale=0.3, size=n) > 0.8).astype(int))


def test_fit_predict_and_contributions_line_up():
    X, y = frame(600, 0)
    m = RiskModel(["phase", "n_countries"], PARAMS, text=TextFeatures(n_components=2, min_df=1)).fit(X, y)
    Xn, _ = frame(50, 1)
    p, c = m.predict_proba(Xn), m.contributions(Xn)
    assert p.shape == (50,) and (0 <= p).all() and (p <= 1).all()
    assert c.shape == (50, len(m.feature_names)) and m.feature_names == ["phase", "n_countries", "txt_00", "txt_01"]


def test_unseen_category_at_scoring_does_not_crash():
    X, y = frame(600, 0)
    m = RiskModel(["phase"], PARAMS, text=None).fit(X, y)
    assert m.categories == {"phase": ["PHASE1", "PHASE2"]}
    assert m.predict_proba(pd.DataFrame({"phase": ["PHASE9", None]})).shape == (2,)


def test_early_stopping_on_a_validation_frame():
    X, y = frame(800, 0)
    m = RiskModel(["phase", "n_countries"], {**PARAMS, "n_estimators": 500}, text=None)
    m.fit(X.iloc[:600], y.iloc[:600], valid=(X.iloc[600:], y.iloc[600:]))
    assert 1 <= m.best_iteration < 500


def test_collapse_text_sums_txt_columns_into_one():
    contrib = np.array([[1.0, 2.0, 0.5, 0.25], [3.0, 4.0, 1.0, 1.0]])
    names = ["phase", "n_countries", "txt_00", "txt_01"]
    out, out_names = collapse_text(contrib, names)
    assert out_names == ["phase", "n_countries", "registration_text"]
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, np.array([[1.0, 2.0, 0.75], [3.0, 4.0, 2.0]]))
