import pytest
from pyspark.sql.types import StringType, StructField, StructType

from ctrisk.spark.clean_trials import build_trials
from ctrisk.spark.trial_attributes import AREAS, build_trial_attributes


@pytest.fixture(scope="module")
def attr_rows(aact):
    trials = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])
    df = build_trial_attributes(trials, aact["countries"], aact["eligibilities"], aact["browse_conditions"])
    return df.collect()


@pytest.fixture(scope="module")
def attrs(attr_rows):
    return {r.nct_id: r for r in attr_rows}


def test_one_row_per_trial(attr_rows):
    nct_ids = [r.nct_id for r in attr_rows]
    assert len(nct_ids) == len(set(nct_ids))
    assert set(nct_ids) == {"NCT001", "NCT002", "NCT003", "NCT009"}


def test_countries_include_removed_ones(attrs):
    assert (attrs["NCT001"].n_countries, attrs["NCT001"].us_only) == (2, False)
    assert (attrs["NCT002"].n_countries, attrs["NCT002"].us_only) == (1, True)
    assert (attrs["NCT009"].n_countries, attrs["NCT009"].us_only) == (0, False)


def test_eligibility(attrs):
    a = attrs["NCT001"]
    assert (a.min_age_years, a.max_age_years, a.healthy_volunteers, a.sex) == (18.0, 75.0, False, "ALL")
    assert a.criteria_count == 3            # headers are not criteria
    b = attrs["NCT003"]
    assert (b.min_age_years, b.max_age_years, b.healthy_volunteers, b.criteria_count) == (0.5, None, True, 0)


def test_disease_areas(attrs):
    assert attrs["NCT001"].area_neoplasms and attrs["NCT001"].area_respiratory
    assert attrs["NCT002"].area_immune and not attrs["NCT002"].area_neoplasms
    assert not any(attrs["NCT009"][f"area_{k}"] for k in AREAS)


def test_criteria_header_variants_without_a_colon_are_not_counted(spark):
    trials = spark.createDataFrame([("NCT100",)], StructType([StructField("nct_id", StringType())]))
    empty_countries = spark.createDataFrame([], StructType([
        StructField("nct_id", StringType()), StructField("name", StringType())]))
    empty_conditions = spark.createDataFrame([], StructType([
        StructField("nct_id", StringType()), StructField("mesh_term", StringType())]))
    eligibilities = spark.createDataFrame(
        [("NCT100", "ALL", None, None, None,
          "Inclusion criteria~1. Adults~Exclusion Criteria :~1. Pregnancy")],
        StructType([
            StructField("nct_id", StringType()),
            StructField("gender", StringType()),
            StructField("minimum_age", StringType()),
            StructField("maximum_age", StringType()),
            StructField("healthy_volunteers", StringType()),
            StructField("criteria", StringType()),
        ]))
    df = build_trial_attributes(trials, empty_countries, eligibilities, empty_conditions)
    row = df.collect()[0]
    assert row.criteria_count == 2
