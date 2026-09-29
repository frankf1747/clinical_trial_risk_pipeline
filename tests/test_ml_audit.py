import numpy as np
import pandas as pd
import pytest

from ctrisk.ml.audit import (
    change_rates,
    compare,
    overlay,
    paired_auc_diff_ci,
    registry_columns,
    unseen_levels,
)


class ScoreIsX:
    """Stands in for a RiskModel: the score is a squashed column x, so swapping x is all that moves the AUC."""
    def __init__(self):
        self.categories = {"phase": ["PHASE1", "PHASE2", "missing"]}

    def predict_proba(self, frame):
        return 1 / (1 + np.exp(-frame["x"].to_numpy(dtype=float)))


def cohort(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    latest = pd.DataFrame({"nct_id": [f"T{i}" for i in range(n)], "label": y,
                           "x": y + rng.normal(scale=0.5, size=n),            # leaks: edited after start
                           "phase": rng.choice(["PHASE1", "PHASE2"], n), "faers_reports": rng.poisson(3, n)})
    pit = latest[["nct_id", "phase"]].copy()
    pit["x"] = rng.normal(size=n)                                             # at start: no signal
    pit["lag_days"] = rng.integers(-60, 30, n)
    return latest, pit


def test_registry_columns_include_disease_areas_but_never_point_in_time_features():
    cols = registry_columns(["phase", "text", "area_skin", "faers_reports", "sponsor_prior_trials", "nct_id"])
    assert cols == ["phase", "text", "area_skin"]


def test_overlay_swaps_registry_columns_and_keeps_the_rest():
    latest = pd.DataFrame({"nct_id": ["A", "B"], "phase": ["PHASE2", "PHASE3"], "faers_reports": [5, 0],
                           "text": ["new", "new"], "label": [1, 0]})
    pit = pd.DataFrame({"nct_id": ["A"], "phase": ["PHASE1"], "text": ["old"]})
    out = overlay(latest, pit, ["phase", "text"])
    assert out.to_dict("records") == [{"nct_id": "A", "phase": "PHASE1", "faers_reports": 5, "text": "old", "label": 1}]


def test_change_rates_split_by_outcome_and_ignore_type_differences():
    latest = pd.DataFrame({"nct_id": list("ABCD"), "label": [1, 1, 0, 0],
                           "n": [3, 3, 1, None], "hv": [True, False, False, None]})
    pit = pd.DataFrame({"nct_id": list("ABCD"), "n": [2.0, 3.0, 1.0, np.nan], "hv": [True, False, False, None]})
    assert change_rates(latest, pit, ["n", "hv"]) == {
        "n": {"terminated": 0.5, "completed": 0.0, "gap": 0.5},
        "hv": {"terminated": 0.0, "completed": 0.0, "gap": 0.0}}


def test_unseen_levels_flag_categories_the_model_never_trained_on():
    pit = pd.DataFrame({"phase": ["PHASE1", "PHASE_1", None, "PHASE2"]})
    assert unseen_levels(pit, {"phase": ["PHASE1", "PHASE2", "missing"]}) == {"phase": 0.3333}


def test_paired_difference_interval_brackets_the_difference():
    latest, pit = cohort()
    y = latest["label"].to_numpy()
    lo, hi = paired_auc_diff_ci(y, latest["x"].to_numpy(), pit["x"].to_numpy(), n=200)
    assert 0 < lo < hi                                          # latest-record x is clearly better


def test_compare_measures_the_drop_on_the_same_trials_and_names_the_leaky_column():
    latest, pit = cohort()
    pit = pit.iloc[:300]                                        # 100 trials have no archived record
    report = compare(ScoreIsX(), latest, pit, ["x", "phase"], bootstrap=100)
    assert (report["n_cohort"], report["n_matched"]) == (400, 300)
    matched = report["matched"]
    assert matched["latest"]["roc_auc"] > 0.8 and matched["point_in_time"]["roc_auc"] < 0.6
    assert matched["auc_drop"] == pytest.approx(matched["latest"]["roc_auc"] - matched["point_in_time"]["roc_auc"], abs=1e-4)
    assert report["swap_one_column"]["x"] > 0.2 and report["swap_one_column"]["phase"] == 0
    assert report["strict"]["n"] == int((pit["lag_days"] <= 0).sum())
    assert report["change_rates"]["phase"]["gap"] == 0.0
