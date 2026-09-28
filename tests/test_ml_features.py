import pandas as pd

from ctrisk.ml.features import inputs, to_matrix

FRAME = pd.DataFrame({
    "nct_id": ["A", "B", "C"],
    "label": [1, 0, None],
    "split": ["train", "train", "score"],
    "start_date": pd.to_datetime(["2010-01-01", "2011-01-01", "2024-01-01"]),
    "phase": ["PHASE2", None, "PHASE1"],
    "n_countries": [3, 1, None],
    "has_faers_history": [True, False, None],
    "faers_reports": [10, 0, 0],
})


def test_inputs_never_include_ids_labels_or_dates():
    assert inputs(FRAME) == ["phase", "n_countries", "has_faers_history", "faers_reports"]
    assert inputs(FRAME, drop=["faers_reports"]) == ["phase", "n_countries", "has_faers_history"]


def test_matrix_fills_missing_categories_and_makes_everything_else_numeric():
    X, categories = to_matrix(FRAME, inputs(FRAME))
    assert list(X["phase"].astype(str)) == ["PHASE2", "missing", "PHASE1"]
    assert categories == {"phase": ["PHASE1", "PHASE2", "missing"]}
    assert X["has_faers_history"].tolist()[:2] == [1.0, 0.0]
    assert X["n_countries"].isna().tolist() == [False, False, True]


def test_scoring_reuses_training_categories():
    _, categories = to_matrix(FRAME, ["phase"])
    new = pd.DataFrame({"phase": ["PHASE3"]})           # unseen at training time
    X, _ = to_matrix(new, ["phase"], categories)
    assert list(X["phase"].cat.categories) == ["PHASE1", "PHASE2", "missing"]
    assert X["phase"].isna().all()                      # unknown -> missing value, not a new code
