import pytest

from ctrisk.gates import DataGateError
from ctrisk.spark.clean_trials import build_drug_interventions, build_trials
from ctrisk.spark.match_drugs import (
    build_trial_drug_map,
    check_match_rate,
    faers_vocab,
    trial_drug_names,
)

VOCAB = frozenset({"pembrolizumab", "metformin", "ibuprofen"})  # adalimumab deliberately absent


@pytest.fixture(scope="module")
def trials(aact):
    return build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])


@pytest.fixture(scope="module")
def names(aact, trials):
    drugs = build_drug_interventions(aact["interventions"], trials)
    return trial_drug_names(drugs, aact["intervention_other_names"], aact["browse_interventions"])


def test_names_come_from_three_sources(names):
    got = {(r.nct_id, r.name) for r in names.collect()}
    assert got == {
        ("NCT001", "Pembrolizumab"), ("NCT001", "MK-3475"),
        ("NCT002", "Adalimumab"),
        ("NCT003", "Metformin 500 mg"),
        ("NCT009", "Ibuprofen"),
        ("NCT010", "Aspirin"),
    }


def test_maps_trials_to_substances(names):
    got = {(r.nct_id, r.substance) for r in build_trial_drug_map(names, VOCAB).collect()}
    assert got == {("NCT001", "pembrolizumab"), ("NCT003", "metformin"), ("NCT009", "ibuprofen")}


def test_match_rate_counts_labeled_trials(trials, names):
    # labeled: NCT001, NCT002, NCT009 -> NCT001 and NCT009 matched
    summary = check_match_rate(trials, build_trial_drug_map(names, VOCAB), min_rate=0.5)
    assert summary == {"labeled_trials": 3, "matched": 2, "match_rate": 0.667, "substances": 3}


def test_match_rate_gate(trials, names):
    with pytest.raises(DataGateError, match="match rate"):
        check_match_rate(trials, build_trial_drug_map(names, VOCAB), min_rate=0.9)


def test_vocab_uses_fda_coded_names_only_and_drops_vague_ones(spark):
    events = spark.createDataFrame([
        ("ondansetron", True, False),     # openFDA standard name
        ("cabazitaxel", False, True),     # FAERS active-substance name
        ("grandma s tonic", False, False),  # free-text product name
        ("vitamin d nos", False, True),   # coded but vague
    ], "substance string, harmonized boolean, active_substance boolean")
    assert faers_vocab(events) == frozenset({"ondansetron", "cabazitaxel"})
