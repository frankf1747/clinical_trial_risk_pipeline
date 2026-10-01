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
    assert set(trials) == {"NCT001", "NCT002", "NCT003", "NCT009", "NCT010"}   # NCT010: finished 2021 start


def test_label_is_terminated_vs_completed_and_null_for_active(trials):
    assert {k: r.label for k, r in trials.items()} == {
        "NCT001": 0, "NCT002": 1, "NCT003": None, "NCT009": 1, "NCT010": 0,
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
        ("NCT010", "Aspirin"),
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


@pytest.mark.parametrize("raw, expected", [
    ("PARALLEL", "PARALLEL"), ("Parallel Assignment", "PARALLEL"),
    ("Single Group Assignment", "SINGLE_GROUP"), ("CROSSOVER", "CROSSOVER"),
    ("N/A", "NA"), ("NA", "NA"), ("Non-Randomized", "NON_RANDOMIZED"),
    ("Supportive Care", "SUPPORTIVE_CARE"), ("Sponsor-Investigator", "SPONSOR_INVESTIGATOR"),
    ("Educational/Counseling/Training", "ECT"), ("ECT", "ECT"),
    (None, None),
])
def test_design_value_maps_legacy_registry_wording_to_current_enums(spark, raw, expected):
    """Archived AACT snapshots predate the 2023 registry modernization; the same answer must map to
    the same category, or the point-in-time audit would read a format change as a record edit."""
    from ctrisk.spark.clean_trials import design_value
    df = spark.createDataFrame([(raw,)], "v string")
    assert df.select(design_value(F.col("v")).alias("r")).first().r == expected


@pytest.mark.parametrize("raw, expected", [
    ("NONE", "NONE"), ("None (Open Label)", "NONE"), ("Open Label", "NONE"),
    ("Double", "DOUBLE"), ("Double Blind (Subject, Investigator)", "DOUBLE"), ("QUADRUPLE", "QUADRUPLE"),
    ("Single Blind (Outcomes Assessor)", "SINGLE"), (None, None),
    # Feb-Aug 2017 archives list the masked parties instead of a level
    ("No masking", "NONE"), ("Participant", "SINGLE"), ("Participant, Investigator", "DOUBLE"),
    ("Participant, Care Provider, Investigator", "TRIPLE"),
    ("Participant, Care Provider, Investigator, Outcomes Assessor", "QUADRUPLE"),
])
def test_masking_value_keeps_only_the_level(spark, raw, expected):
    from ctrisk.spark.clean_trials import masking_value
    df = spark.createDataFrame([(raw,)], "v string")
    assert df.select(masking_value(F.col("v")).alias("r")).first().r == expected


@pytest.mark.parametrize("raw, expected", [
    ("t", True), ("f", False), ("true", True), ("Yes", True), ("No", False),
    ("Accepts Healthy Volunteers", True), ("", None), (None, None),
])
def test_flag_reads_every_boolean_spelling(spark, raw, expected):
    from ctrisk.spark.clean_trials import flag
    df = spark.createDataFrame([(raw,)], "v string")
    assert df.select(flag(F.col("v")).alias("r")).first().r == expected


def test_study_fields_keeps_every_study_unfiltered(aact):
    from ctrisk.spark.clean_trials import study_fields
    rows = {r.nct_id: r for r in study_fields(aact["studies"], aact["designs"], aact["sponsors"]).collect()}
    assert len(rows) == 10                                   # observational, withdrawn, Phase 4 included
    assert (rows["NCT004"].study_type, rows["NCT007"].status) == ("OBSERVATIONAL", "WITHDRAWN")
    assert (rows["NCT001"].masking, rows["NCT001"].has_dmc, rows["NCT001"].sponsor_class) == ("DOUBLE", True, "INDUSTRY")
    assert rows["NCT003"].start_date_type == "ANTICIPATED"


@pytest.mark.parametrize("raw, parties, expected", [
    ("Double Blind", ("t", "t", "t", "t"), "QUADRUPLE"),     # 2017 wording: "double blind" meant any blinding
    ("Double Blind", ("t", "f", "t", "f"), "DOUBLE"),
    ("Double Blind", ("t", "t", "t", "f"), "TRIPLE"),
    ("Single Blind", ("f", "f", "f", "t"), "SINGLE"),
    ("Double-Blind", ("f", "f", "f", "f"), "DOUBLE"),         # no parties recorded: keep the wording
    ("Open Label", ("f", "f", "f", "f"), "NONE"),
    ("DOUBLE", ("t", "t", "t", "t"), "DOUBLE"),               # current enum already counts parties
    (None, (None, None, None, None), None),
])
def test_masking_value_counts_masked_parties_for_legacy_blind_wording(spark, raw, parties, expected):
    """Before the 2017 final rule, 'Double Blind' covered 2-4 masked parties; the current enum counts them."""
    from ctrisk.spark.clean_trials import MASKED_PARTIES, masked_parties, masking_value
    df = spark.createDataFrame([(raw, *parties)], "v string, " + ", ".join(f"{c} string" for c in MASKED_PARTIES))
    assert df.select(masking_value(F.col("v"), masked_parties(df.columns)).alias("r")).first().r == expected


def test_study_fields_read_a_2017_archive_like_a_current_one(spark):
    """Archived snapshots: start_month_year instead of start_date, blank allocation for single-arm trials,
    'U.S. Fed' sponsors. Each must land on the value the current snapshot would give."""
    from ctrisk.spark.clean_trials import study_fields
    studies = spark.createDataFrame(
        [("NCT1", "Interventional", "Completed", "Phase 2", "January 2015", "1", "", "", "t"),
         ("NCT2", "Interventional", "Terminated", "Phase 1", "March 3, 2016", "2", "", "", "f")],
        "nct_id string, study_type string, overall_status string, phase string, start_month_year string, "
        "number_of_arms string, brief_title string, why_stopped string, has_dmc string")
    designs = spark.createDataFrame(
        [("NCT1", None, "Single Group Assignment", "Treatment", "Open Label", "f", "f", "f", "f"),
         ("NCT2", None, "Parallel Assignment", "Treatment", "Double Blind", "t", "t", "t", "t")],
        "nct_id string, allocation string, intervention_model string, primary_purpose string, masking string, "
        "subject_masked string, caregiver_masked string, investigator_masked string, outcomes_assessor_masked string")
    sponsors = spark.createDataFrame([("NCT1", "lead", "VA", "U.S. Fed"), ("NCT2", "lead", "Acme", "Industry")],
                                     "nct_id string, lead_or_collaborator string, name string, agency_class string")
    rows = {r.nct_id: r for r in study_fields(studies, designs, sponsors).collect()}
    assert (str(rows["NCT1"].start_date), str(rows["NCT2"].start_date)) == ("2015-01-31", "2016-03-03")
    assert (rows["NCT1"].allocation, rows["NCT2"].allocation) == ("NA", None)      # only single-arm blanks are NA
    assert (rows["NCT1"].sponsor_class, rows["NCT2"].masking) == ("GOVERNMENT", "QUADRUPLE")
