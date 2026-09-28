import numpy as np

from ctrisk.ml.score import risk_deciles, top_drivers


def test_top_drivers_are_the_largest_contributions_toward_termination():
    contrib = np.array([[0.5, -0.9, 0.1, 0.3],
                        [-0.2, 0.4, 0.8, 0.0]])
    assert top_drivers(contrib, ["a", "b", "c", "d"], k=2) == [["a", "d"], ["c", "b"]]


def test_deciles_run_from_1_lowest_to_10_highest():
    d = risk_deciles(np.linspace(0, 1, 100))
    assert (d.min(), d.max()) == (1, 10)
    assert (d == 10).sum() == 10 and d[-1] == 10


def test_drivers_only_list_features_that_raise_risk():
    contrib = np.array([[0.5, -0.9, -0.1, -0.3]])
    assert top_drivers(contrib, ["a", "b", "c", "d"]) == [["a", None, None]]
