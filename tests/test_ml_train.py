import numpy as np
import pandas as pd
import pytest

from ctrisk.ml.train import run

SMALL_GRID = [{"num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 20}]


def synthetic(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    frame = pd.DataFrame({
        "nct_id": [f"NCT{i:05d}" for i in range(n)],
        "split": rng.choice(["train", "test", "recent", "score"], n, p=[0.55, 0.2, 0.15, 0.1]),
        "start_date": pd.to_datetime(rng.choice(pd.date_range("2008-01-01", "2014-12-31", freq="D"), n)),
        "phase": rng.choice(["PHASE1", "PHASE2", "PHASE3"], n),
        "sponsor_class": rng.choice(["INDUSTRY", "OTHER"], n),
        "n_countries": signal,
        "faers_reports": rng.poisson(3, n),                       # noise
        "text": [("slow accrual " if s > 1.5 else "large multicenter ") + rng.choice(["a", "b"]) for s in signal],
    })
    terminated = (signal + rng.normal(scale=0.5, size=n) > 1).astype(float)
    frame["label"] = np.where(frame["split"] == "score", np.nan, terminated)
    reason = rng.choice(["enrollment", "safety", "business"], n, p=[0.6, 0.2, 0.2])
    frame["label_enrollment"] = np.where(frame["label"] == 0, 0.0,
                                         np.where((frame["label"] == 1) & (reason == "enrollment"), 1.0, np.nan))
    frame["label_safety"] = np.where(frame["label"] == 0, 0.0,
                                     np.where((frame["label"] == 1) & (reason == "safety"), 1.0, np.nan))
    return frame


@pytest.fixture(scope="module")
def result():
    return run(synthetic(), grid=SMALL_GRID, text_min_df=1)


def test_every_target_is_trained_and_evaluated(result):
    _, report = result
    assert set(report["targets"]) == {"label", "label_enrollment", "label_safety"}
    label = report["targets"]["label"]
    assert set(label["models"]) == {"logistic_regression", "lightgbm", "lightgbm_no_text",
                                    "lightgbm_no_faers", "lightgbm_no_burden"}
    assert label["models"]["lightgbm"]["test"]["roc_auc"] > 0.85
    lo, hi = label["models"]["lightgbm"]["test"]["roc_auc_ci95"]
    assert lo < label["models"]["lightgbm"]["test"]["roc_auc"] < hi
    assert label["params"]["n_estimators"] >= 50 and len(label["grid"]) == 1


def test_reason_targets_exclude_other_terminations(result):
    _, report = result
    enrol = report["targets"]["label_enrollment"]["splits"]["train"]["n"]
    all_ = report["targets"]["label"]["splits"]["train"]["n"]
    assert 0 < enrol < all_


def test_ablations_and_subgroups(result):
    _, report = result
    label = report["targets"]["label"]
    assert label["models"]["lightgbm_no_burden"]["test"]["roc_auc"] < 0.8   # n_countries is the driver
    assert set(label["by_sponsor_class"]) == {"INDUSTRY", "OTHER"}
    assert set(label["by_phase"]) == {"PHASE1", "PHASE2", "PHASE3"}
    assert all(g["n"] >= 100 for g in label["by_start_year"].values()) and label["by_start_year"]
    fit = label["models"]["lightgbm"]["test"]["calibration_fit"]
    assert set(fit) == {"intercept", "slope"}
    assert report["top_drivers"][0]["feature"] == "n_countries"


def test_returns_scoring_models_for_overall_and_enrollment(result):
    models, report = result
    assert set(models) == {"label", "label_enrollment"}
    assert report["features"]["columns"] == ["phase", "sponsor_class", "n_countries", "faers_reports"]
    assert models["label"].feature_names[:4] == report["features"]["columns"]


def test_refuses_a_train_test_or_recent_row_without_a_label():
    frame = synthetic(200)
    frame.loc[frame["split"] == "train", "label"] = np.nan
    with pytest.raises(ValueError, match="without a label"):
        run(frame, grid=SMALL_GRID, text_min_df=1)


def test_reason_targets_count_and_evaluate_only_their_own_rows(result):
    _, report = result
    frame = synthetic()
    for target in ("label_enrollment", "label_safety"):
        for split in ("train", "test", "recent"):
            rows = frame[(frame["split"] == split) & frame[target].notna()]
            got = report["targets"][target]["splits"][split]
            assert got["n"] == len(rows) and got["base_rate"] == round(float(rows[target].mean()), 4)
            if split != "train":
                assert report["targets"][target]["models"]["lightgbm"][split]["n"] == len(rows)


def test_text_is_only_ever_fit_on_training_rows(monkeypatch):
    from ctrisk.ml.text import TextFeatures
    frame = synthetic(1500)
    frame["text"] = frame["nct_id"]                    # each document names its own trial
    seen, original = [], TextFeatures.fit
    monkeypatch.setattr(TextFeatures, "fit", lambda self, texts: seen.append(set(texts)) or original(self, texts))
    run(frame, grid=SMALL_GRID, text_min_df=1)
    train = frame["split"] == "train"
    assert seen and all(s <= set(frame.loc[train, "nct_id"]) for s in seen)
    inner = train & (frame["start_date"] < "2013-01-01") & frame["label"].notna()
    assert seen[0] == set(frame.loc[inner, "nct_id"])  # tuning: inner-train rows only


def test_top_drivers_say_which_way_each_feature_pushes(result):
    _, report = result
    top = {d["feature"]: d for d in report["top_drivers"]}
    n = top["n_countries"]["direction"]                       # the synthetic signal: more -> terminated
    assert n["high_third"] > 0 > n["low_third"]
    assert set(top["phase"]["direction"]) <= {"PHASE1", "PHASE2", "PHASE3", "missing"}
    assert top["registration_text"]["direction"] is None      # a composite has no single value


def test_driver_directions_treat_booleans_as_levels():
    from ctrisk.ml.train import driver_directions
    frame = pd.DataFrame({"hv": [True, False, None, False], "x": [1.0, 2.0, 3.0, np.nan]})
    contrib = np.array([[-0.4, -0.1], [0.2, 0.0], [0.0, 0.3], [0.4, 0.0]])
    d = driver_directions(contrib, ["hv", "x", "registration_text"][:2], frame)
    assert d["hv"] == {"False": 0.3, "True": -0.4, "missing": 0.0}
    assert d["x"] == {"low_third": -0.1, "high_third": 0.3}
