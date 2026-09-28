import numpy as np
import pandas as pd
import pytest

from ctrisk.ml.train import run


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(0)
    n = 4000
    signal = rng.normal(size=n)                          # the only real driver
    frame = pd.DataFrame({
        "nct_id": [f"NCT{i:05d}" for i in range(n)],
        "split": rng.choice(["train", "test", "recent", "score"], n, p=[0.55, 0.2, 0.15, 0.1]),
        "start_date": pd.Timestamp("2012-01-01"),
        "phase": rng.choice(["PHASE1", "PHASE2", "PHASE3"], n),
        "n_countries": signal,
        "faers_reports": rng.poisson(3, n),              # noise
    })
    frame["label"] = np.where(frame["split"] == "score", np.nan,
                              (signal + rng.normal(scale=0.5, size=n) > 1).astype(float))
    return run(frame)


def test_reports_every_model_on_test_and_recent(result):
    _, report = result
    assert set(report["models"]) == {"logistic_regression", "lightgbm", "lightgbm_no_faers", "lightgbm_no_burden"}
    for m in report["models"].values():
        assert set(m) == {"test", "recent"}
    assert report["models"]["lightgbm"]["test"]["roc_auc"] > 0.85
    lo, hi = report["models"]["lightgbm"]["test"]["roc_auc_ci95"]
    assert lo < report["models"]["lightgbm"]["test"]["roc_auc"] < hi


def test_ablation_without_the_real_driver_collapses(result):
    _, report = result
    assert report["models"]["lightgbm_no_burden"]["test"]["roc_auc"] < 0.65   # n_countries is in BURDEN


def test_top_driver_is_the_real_signal(result):
    _, report = result
    assert report["top_drivers"][0]["feature"] == "n_countries"


def test_returns_the_full_model_with_its_inputs(result):
    model, report = result
    assert report["features"]["columns"] == ["phase", "n_countries", "faers_reports"]
    assert hasattr(model, "predict_proba")


def test_refuses_a_train_test_or_recent_row_without_a_label():
    frame = pd.DataFrame({"nct_id": ["A", "B"], "split": ["train", "test"], "label": [1.0, np.nan],
                          "start_date": pd.Timestamp("2012-01-01"), "n_countries": [1.0, 2.0]})
    with pytest.raises(ValueError, match="without a label"):
        run(frame)
