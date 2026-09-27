"""AACT -> one row per eligible drug trial, with its label.

Population: interventional drug/biologic trials, Phase 1-3.
  Finished (completed/terminated) and started 2008-2020 -> label 0/1, used for training.
  Still active                                         -> label null, scored later.
"""
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from ctrisk.config import load_config
from ctrisk.gates import DataGateError
from ctrisk.spark.session import get_spark

PHASES = ["PHASE1", "PHASE1/PHASE2", "PHASE2", "PHASE2/PHASE3", "PHASE3"]
ACTIVE = ["NOT_YET_RECRUITING", "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]
DRUG_TYPES = ["DRUG", "BIOLOGICAL"]
GOVERNMENT = ["NIH", "FED", "OTHER_GOV"]
DESIGN_COLS = ["allocation", "intervention_model", "primary_purpose", "masking"]
TRAIN_START, TRAIN_END = "2008-01-01", "2020-12-31"


def read_table(spark: SparkSession, aact_dir: str, name: str) -> DataFrame:
    return spark.read.csv(f"{aact_dir}/{name}.txt", sep="|", header=True,
                          multiLine=True, quote='"', escape='"')


def norm(col: Column) -> Column:
    """'Active, not recruiting' -> 'ACTIVE_NOT_RECRUITING'; 'Phase 1/Phase 2' -> 'PHASE1/PHASE2'."""
    c = F.regexp_replace(F.upper(F.trim(col)), r"PHASE\s+", "PHASE")
    return F.regexp_replace(c, r"[^A-Z0-9/]+", "_")


def build_trials(studies: DataFrame, designs: DataFrame, sponsors: DataFrame,
                 interventions: DataFrame) -> DataFrame:
    drug_trials = (interventions
                   .where(norm(F.col("intervention_type")).isin(DRUG_TYPES))
                   .select("nct_id").distinct())

    agency = norm(F.col("agency_class"))
    lead_sponsor = (sponsors
                    .where(F.lower("lead_or_collaborator") == "lead")
                    .select("nct_id",
                            F.col("name").alias("sponsor_name"),
                            F.when(agency == "INDUSTRY", "INDUSTRY")
                             .when(agency.isin(GOVERNMENT), "GOVERNMENT")
                             .otherwise("OTHER").alias("sponsor_class"))
                    .dropDuplicates(["nct_id"]))

    design = designs.select("nct_id", *[norm(F.col(c)).alias(c) for c in DESIGN_COLS])

    s = studies.select(
        "nct_id",
        norm(F.col("study_type")).alias("study_type"),
        norm(F.col("overall_status")).alias("status"),
        norm(F.col("phase")).alias("phase"),
        F.to_date("start_date").alias("start_date"),
        F.col("number_of_arms").cast("int").alias("number_of_arms"),
    )
    finished = (F.col("status").isin("COMPLETED", "TERMINATED")
                & F.col("start_date").between(TRAIN_START, TRAIN_END))
    eligible = (F.col("study_type") == "INTERVENTIONAL") & F.col("phase").isin(PHASES) \
        & (finished | F.col("status").isin(ACTIVE))

    return (s.where(eligible)
            .join(drug_trials, "nct_id")
            .withColumn("label", F.when(F.col("status") == "TERMINATED", 1)
                                  .when(F.col("status") == "COMPLETED", 0).cast("int"))
            .join(design, "nct_id", "left")
            .join(lead_sponsor, "nct_id", "left")
            .drop("study_type"))


def build_drug_interventions(interventions: DataFrame, trials: DataFrame) -> DataFrame:
    return (interventions
            .where(norm(F.col("intervention_type")).isin(DRUG_TYPES))
            .join(trials.select("nct_id"), "nct_id", "left_semi")
            .select("nct_id", F.col("id").alias("intervention_id"), "name"))


def build_sponsor_outcomes(studies: DataFrame, sponsors: DataFrame) -> DataFrame:
    """Every finished interventional study, with its lead sponsor and end date.

    This is the history sponsor features look back on; Snowflake only uses rows that
    ended before a given trial started.
    """
    status = norm(F.col("overall_status"))
    lead = (sponsors.where(F.lower("lead_or_collaborator") == "lead")
            .select("nct_id", F.col("name").alias("sponsor_name"))
            .dropDuplicates(["nct_id"]))
    return (studies
            .where((norm(F.col("study_type")) == "INTERVENTIONAL")
                   & status.isin("COMPLETED", "TERMINATED")
                   & F.col("completion_date").isNotNull())
            .select("nct_id",
                    (status == "TERMINATED").cast("int").alias("terminated"),
                    F.to_date("completion_date").alias("completion_date"))
            .join(lead, "nct_id"))


def check_trials(trials: DataFrame, min_trials: int) -> dict:
    labeled, rate, active = trials.agg(
        F.count("label"), F.avg("label"), F.sum(F.col("label").isNull().cast("int"))
    ).first()
    if not labeled or labeled < min_trials:
        raise DataGateError(f"{labeled or 0} labeled trials, expected at least {min_trials}")
    if not 0.05 <= rate <= 0.25:
        raise DataGateError(f"termination rate {rate:.1%} is outside 5-25%; check the label logic")
    return {"labeled": labeled, "termination_rate": round(rate, 3), "active": active}


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("clean_trials", cfg.mode)
    t = {name: read_table(spark, cfg.path("raw", "aact"), name)
         for name in ("studies", "designs", "sponsors", "interventions")}

    trials = build_trials(t["studies"], t["designs"], t["sponsors"], t["interventions"]).cache()
    summary = check_trials(trials, cfg.min_trials)

    trials.write.mode("overwrite").parquet(cfg.path("parquet", "trials"))
    build_drug_interventions(t["interventions"], trials) \
        .write.mode("overwrite").parquet(cfg.path("parquet", "drug_interventions"))
    build_sponsor_outcomes(t["studies"], t["sponsors"]) \
        .write.mode("overwrite").parquet(cfg.path("parquet", "sponsor_outcomes"))
    print(summary)
