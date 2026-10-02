import math

import pytest

from ctrisk.serving.labels import reason_text, value_text


@pytest.mark.parametrize("feature, value, shown", [
    ("healthy_volunteers", False, "Accepts healthy volunteers: No"),
    ("phase", "PHASE1/PHASE2", "Phase: 1/2"),
    ("sponsor_class", "OTHER", "Sponsor type: academic or other"),
    ("sponsor_prior_termination_rate", 0.5, "Sponsor's past termination rate: 50%"),
    ("sponsor_prior_trials", 12.0, "Sponsor's past trials: 12"),
    ("intervention_model", "SINGLE_GROUP", "Design: single group"),
    ("masking", None, "Blinding: not given"),
    ("max_age_years", math.nan, "Maximum age: not given"),
    ("area_neoplasms", True, "Disease area, cancer: Yes"),
    ("faers_death_share", 0.0417, "Drug's FAERS reports with a death: 4.2%"),
    ("registration_text", None, "Wording of the registration (title, summary, criteria)"),
])
def test_value_text_reads_like_a_sentence(feature, value, shown):
    assert value_text(feature, value) == shown


def test_reason_text_says_which_way():
    assert reason_text("healthy_volunteers", False, 0.19) == "Accepts healthy volunteers: No (raises risk)"
    assert reason_text("phase", "PHASE1", -0.12) == "Phase: 1 (lowers risk)"


def test_every_model_feature_has_a_label():
    import json
    from pathlib import Path

    from ctrisk.serving.labels import LABELS
    features = json.loads((Path(__file__).resolve().parents[1] / "models" / "v4" / "features.json").read_text())
    assert set(features["columns"]) <= set(LABELS)
