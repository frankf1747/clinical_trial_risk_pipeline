"""One text blob per trial from fields written at registration: title, summary, eligibility
criteria, keywords. These are rarely rewritten after a trial starts.

Deliberately absent: design_outcomes. AACT rewrites outcome measures when results are
posted, well after the trial starts and correlated with how it ended, so they would leak.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ctrisk.config import load_config
from ctrisk.spark.clean_trials import read_table
from ctrisk.spark.session import get_spark

# A summary sentence that mentions the trial stopping early gives the label away; strip it
# (and only it) out, keeping the rest of the summary.
TERMINATION_WORDING = r"(?i)[^.~]*\b(terminat|prematurely|halted|stopped early|was discontinued|enrollment was closed)[^.~]*[.~]?"


def build_trial_text(trials: DataFrame, studies: DataFrame, brief_summaries: DataFrame,
                     eligibilities: DataFrame, keywords: DataFrame) -> DataFrame:
    kw = keywords.groupBy("nct_id").agg(
        F.concat_ws(", ", F.array_sort(F.collect_list("name"))).alias("keywords"))
    parts = [F.coalesce("official_title", "brief_title"), F.col("summary"),
             F.regexp_replace(F.col("criteria"), "~", " "), F.col("keywords")]
    return (trials.select("nct_id")
            .join(studies.select("nct_id", "official_title", "brief_title"), "nct_id", "left")
            .join(brief_summaries.select(
                "nct_id", F.regexp_replace(F.col("description"), TERMINATION_WORDING, " ").alias("summary")),
                "nct_id", "left")
            .join(eligibilities.select("nct_id", "criteria"), "nct_id", "left")
            .join(kw, "nct_id", "left")
            .select("nct_id", F.concat_ws("\n", *parts).alias("text")))   # concat_ws skips nulls


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("trial_text", cfg.mode)
    aact = cfg.path("raw", "aact")
    text = build_trial_text(spark.read.parquet(cfg.path("parquet", "trials")),
                            *[read_table(spark, aact, t) for t in
                              ("studies", "brief_summaries", "eligibilities", "keywords")])
    text.write.mode("overwrite").parquet(cfg.path("parquet", "trial_text"))
    print({"trials": text.count(), "median_chars": text.select(F.expr("percentile(length(text), 0.5)")).first()[0]})
