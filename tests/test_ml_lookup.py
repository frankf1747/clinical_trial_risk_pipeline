import json

import numpy as np
import pandas as pd
import pytest
from test_ml_train import synthetic

from ctrisk.ml.lookup import cross_fit, lookup_rows, percentile, score_all
from ctrisk.ml.model import RiskModel
from ctrisk.ml.text import TextFeatures

PARAMS = {"n_estimators": 60, "num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 20,
          "random_state": 0, "verbose": -1}
COLUMNS = ["phase", "sponsor_class", "n_countries", "faers_reports"]


@pytest.fixture(scope="module")
def frame():
    f = synthetic(3000)
    f["split"] = np.where(pd.to_datetime(f["start_date"]).dt.year >= 2013,
                          np.where(f["split"] == "score", "score", "test"), f["split"].replace({"test": "train", "recent": "train"}))
    return f


@pytest.fixture(scope="module")
def model(frame):
    train = frame[(frame["split"] == "train") & frame["label"].notna()]
    return RiskModel(COLUMNS, PARAMS, text=TextFeatures(min_df=1)).fit(train, train["label"])


def test_cross_fit_never_scores_a_trial_with_a_model_that_saw_its_year(frame, model, monkeypatch):
    seen = []
    original = RiskModel.fit
    monkeypatch.setattr(RiskModel, "fit", lambda self, f, y, valid=None: seen.append(set(f["nct_id"])) or original(self, f, y))
    p, contrib, names = cross_fit(model, frame, frame["label"])
    train = frame[frame["split"] == "train"]
    years = pd.to_datetime(train["start_date"]).dt.year
    assert len(seen) == years.nunique()
    for fitted, year in zip(seen, sorted(years.unique())):
        assert not fitted & set(train.loc[years == year, "nct_id"])         # held-out year never in the fit
    assert set(p.index) == set(train.index) and p.notna().all()
    assert contrib.shape == (len(train), len(names)) and names[-1] == "registration_text"


def test_score_all_labels_where_each_score_came_from(frame, model):
    out = score_all(model, frame, frame["label"])
    kinds = dict(zip(frame["split"], out.loc[frame.index, "score_type"]))
    assert kinds == {"train": "out_of_fold", "test": "held_out", "score": "forward"}
    assert out["score"].between(0, 1).all()
    held = frame["split"] != "train"
    assert np.allclose(out.loc[held, "score"], model.predict_proba(frame[held]))   # final model, unchanged


def test_percentile_is_the_share_of_active_trials_scored_lower():
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    assert percentile(np.array([0.05, 0.25, 0.4, 0.9]), ref).tolist() == [0.0, 50.0, 75.0, 100.0]


def test_lookup_rows_carry_percentile_enrollment_risk_and_plain_reasons(frame, model):
    overall = score_all(model, frame, frame["label"])
    rows = lookup_rows(frame, overall, overall, version=4)
    assert list(rows.columns) == ["NCT_ID", "MODEL_VERSION", "SCORE_TYPE", "RISK_SCORE", "RISK_PERCENTILE",
                                  "ENROLLMENT_RISK_SCORE", "REASONS"]
    assert rows["MODEL_VERSION"].eq("v4").all() and rows["RISK_PERCENTILE"].between(0, 100).all()
    reasons = json.loads(rows["REASONS"].iloc[0])
    assert 1 <= len(reasons) <= 5 and {"feature", "text", "contribution"} <= set(reasons[0])
    ups = [r["contribution"] for r in reasons if r["contribution"] > 0]
    assert ups == sorted(ups, reverse=True)                                  # strongest first
    assert all(("raises risk" in r["text"]) == (r["contribution"] > 0) for r in reasons)
