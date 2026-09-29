from pyspark.sql.types import StringType, StructField, StructType

from ctrisk.spark.clean_trials import build_trials
from ctrisk.spark.trial_text import build_trial_text


def test_one_blob_per_trial_from_registration_fields(aact):
    trials = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])
    rows = {r.nct_id: r.text for r in build_trial_text(
        trials, aact["studies"], aact["brief_summaries"], aact["eligibilities"], aact["keywords"]).collect()}

    assert set(rows) == {"NCT001", "NCT002", "NCT003", "NCT009", "NCT010"}
    t = rows["NCT001"]
    assert "A Phase 2 Study of Pembrolizumab" in t          # official title preferred
    assert "advanced lung cancer" in t                       # summary
    assert "Measurable disease" in t and "~" not in t        # criteria, line breaks removed
    assert "Overall survival" not in t                       # design_outcomes are rewritten post-start: dropped
    assert "PD-1" in t                                       # keywords
    assert rows["NCT002"].startswith("Adalimumab in RA")     # falls back to brief title
    assert rows["NCT009"] == "Ibuprofen Dosing"              # nothing else registered


def test_termination_wording_scrubbed_from_summary(spark):
    trials = spark.createDataFrame([("NCT200",)], StructType([StructField("nct_id", StringType())]))
    studies = spark.createDataFrame(
        [("NCT200", None, "Some Trial")],
        StructType([StructField("nct_id", StringType()), StructField("official_title", StringType()),
                    StructField("brief_title", StringType())]))
    summaries = spark.createDataFrame(
        [("NCT200", "Drug X in adults. The study was terminated early due to business reasons.")],
        StructType([StructField("nct_id", StringType()), StructField("description", StringType())]))
    empty_elig = spark.createDataFrame([], StructType([
        StructField("nct_id", StringType()), StructField("criteria", StringType())]))
    empty_keywords = spark.createDataFrame([], StructType([
        StructField("nct_id", StringType()), StructField("name", StringType())]))

    text = build_trial_text(trials, studies, summaries, empty_elig, empty_keywords).collect()[0].text
    assert "Drug X in adults" in text
    assert "terminated" not in text.lower()
    assert "business reasons" not in text
