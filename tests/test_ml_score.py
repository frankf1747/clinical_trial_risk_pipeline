import numpy as np
import pandas as pd

from ctrisk.ml.score import describe, risk_deciles, top_drivers

NAMES = ["a", "b", "c", "d"]
FRAME = pd.DataFrame({"a": [True, False], "b": [1.0, 2.0], "c": ["PHASE2", "PHASE3"], "d": [None, 3.5]})


def test_top_drivers_are_the_largest_contributions_toward_termination_with_their_values():
    contrib = np.array([[0.5, -0.9, 0.1, 0.3],
                        [-0.2, 0.4, 0.8, 0.0]])
    assert top_drivers(contrib, NAMES, FRAME, k=2) == [[("a=Yes", 0.5), ("d=missing", 0.3)],
                                                       [("c=PHASE3", 0.8), ("b=2", 0.4)]]


def test_drivers_only_list_features_that_raise_risk():
    contrib = np.array([[0.5, -0.9, -0.1, -0.3]])
    assert top_drivers(contrib, NAMES, FRAME.iloc[:1]) == [[("a=Yes", 0.5), None, None]]


def test_describe_shows_booleans_as_yes_no_and_text_as_a_composite():
    assert describe("healthy_volunteers", False) == "healthy_volunteers=No"
    assert describe("healthy_volunteers", np.bool_(True)) == "healthy_volunteers=Yes"
    assert describe("n_countries", 12.0) == "n_countries=12"
    assert describe("faers_death_share", 0.0417) == "faers_death_share=0.0417"
    assert describe("phase", None) == "phase=missing"
    assert describe("registration_text", float("nan")) == "registration_text (net of all text components)"


def test_drivers_read_values_from_the_scored_frame_not_its_position():
    frame = FRAME.set_index(pd.Index([700, 701]))               # a slice of a larger frame
    contrib = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0]])
    assert top_drivers(contrib, NAMES, frame, k=1)[1] == [("c=PHASE3", 0.2)]


def test_deciles_run_from_1_lowest_to_10_highest():
    d = risk_deciles(np.linspace(0, 1, 100))
    assert (d.min(), d.max()) == (1, 10)
    assert (d == 10).sum() == 10 and d[-1] == 10
