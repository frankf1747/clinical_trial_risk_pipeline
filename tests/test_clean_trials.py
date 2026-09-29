import pytest
from pyspark.sql import functions as F

from ctrisk.gates import DataGateError
from ctrisk.spark.clean_trials import (
    build_drug_interventions,
    build_sponsor_starts,
    build_trials,
    check_trials,
)


@pytest.fixture(scope="module")
def trials(aact):
    rows = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"]).collect()
    return {r.nct_id: r for r in rows}


def test_keeps_only_eligible_drug_trials(trials):
    assert set(trials) == {"NCT001", "NCT002", "NCT003", "NCT009"}


def test_label_is_terminated_vs_completed_and_null_for_active(trials):
    assert {k: r.label for k, r in trials.items()} == {
        "NCT001": 0, "NCT002": 1, "NCT003": None, "NCT009": 1,
    }


def test_normalizes_legacy_casing(trials):
    assert trials["NCT009"].phase == "PHASE1/PHASE2"
    assert trials["NCT009"].status == "TERMINATED"
    assert trials["NCT009"].sponsor_class == "INDUSTRY"


def test_uses_lead_sponsor_only(trials):
    assert trials["NCT002"].sponsor_class == "OTHER"
    assert trials["NCT002"].sponsor_name == "State University"


def test_maps_nih_to_government(trials):
    assert trials["NCT003"].sponsor_class == "GOVERNMENT"


def test_missing_design_is_null_not_dropped(trials):
    assert trials["NCT009"].masking is None
    assert trials["NCT001"].masking == "DOUBLE"


def test_drug_interventions_only_for_eligible_trials(aact):
    trials = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])
    rows = build_drug_interventions(aact["interventions"], trials).collect()
    assert sorted((r.nct_id, r.name) for r in rows) == [
        ("NCT001", "Pembrolizumab"),
        ("NCT002", "Adalimumab"),
        ("NCT003", "Metformin 500 mg"),
        ("NCT009", "Ibuprofen"),
    ]


def test_gate_fails_below_minimum_trials(aact):
    trials = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])
    with pytest.raises(DataGateError, match="labeled trials"):
        check_trials(trials, min_trials=100)


def test_gate_fails_on_implausible_termination_rate(aact):
    # Fixture: 2 of 3 labeled trials terminated (67%), outside 5-25%
    trials = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])
    with pytest.raises(DataGateError, match="termination rate"):
        check_trials(trials, min_trials=1)


def test_gate_passes_and_summarizes(spark):
    rows = [(f"NCT{i:03d}", 1 if i == 0 else 0) for i in range(10)] + [
        ("NCT100", None), ("NCT101", None),
    ]
    df = spark.createDataFrame(rows, "nct_id string, label int")
    assert check_trials(df, min_trials=10) == {
        "labeled": 10, "termination_rate": 0.1, "active": 2,
    }


def test_stop_reason_only_for_terminated_trials(trials):
    assert trials["NCT002"].stop_reason == "enrollment"
    assert trials["NCT009"].stop_reason == "business"
    assert trials["NCT001"].stop_reason is None


def test_has_dmc(trials):
    assert (trials["NCT001"].has_dmc, trials["NCT002"].has_dmc) == (True, False)


@pytest.mark.parametrize("text, expected", [
    ("Slow accrual", "enrollment"),
    ("Sponsor decision due to slow enrollment", "enrollment"),      # root cause wins over "sponsor"
    ("Unable to recruit eligible patients", "enrollment"),
    ("Company strategic decision; portfolio prioritization", "business"),
    ("Funding ended", "business"),
    ("Unacceptable toxicity in the first cohort", "safety"),
    ("Terminated for futility at interim analysis", "efficacy"),
    ("Lack of efficacy", "efficacy"),
    ("Safety signal and low enrollment", "safety"),                   # safety outranks enrollment
    ("PI left the institution", "other"),
    ("", None),
    ("Business decision, not related to safety", "business"),
    ("Lack of efficacy; no safety concern", "efficacy"),
    ("Dose limiting toxicities", "safety"),
    # "harm" is inside "pharmaceutical" but \b keeps it from matching mid-word, so this stays "other"
    ("Lack of support from pharmaceutical collaborator", "other"),
    ("Slow enrollment, interim analysis conducted", "enrollment"),
])
def test_stop_reason_classifier(spark, text, expected):
    from ctrisk.spark.clean_trials import stop_reason
    df = spark.createDataFrame([(text,)], "why_stopped string")
    assert df.select(stop_reason(F.col("why_stopped")).alias("r")).first().r == expected


def test_sponsor_starts_excludes_withdrawn_observational_and_sponsorless(aact):
    rows = {r.nct_id: (r.sponsor_name, str(r.start_date))
            for r in build_sponsor_starts(aact["studies"], aact["sponsors"]).collect()}
    # NCT007 (withdrawn) and NCT004 (observational) each have a lead sponsor in the fixture,
    # so their exclusion here proves the status/study_type filters, not just the join.
    assert set(rows) == {"NCT001", "NCT002", "NCT003", "NCT009"}
    assert rows["NCT002"] == ("State University", "2015-06-15")


def test_brief_title_is_carried_for_display(trials):
    assert trials["NCT001"].brief_title == "Pembrolizumab in Lung Cancer"
