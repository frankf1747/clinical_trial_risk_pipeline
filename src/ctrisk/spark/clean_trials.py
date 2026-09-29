"""AACT -> one row per eligible drug trial, with its label.

Population: interventional drug/biologic trials, Phase 1-3.
  Finished (completed/terminated), started 2008 or later -> label 0/1. Training uses 2008-2020
                                                           starts; later ones feed the backtest.
  Still active                                          -> label null, scored.
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
TRAIN_START = "2008-01-01"
MODEL_END = "2021-01-01"     # modelling uses starts before this; later finished trials feed the backtest

# Why a trial stopped, from the free-text reason. First match wins, so a specific cause
# ("safety") outranks a generic one ("sponsor decision").
STOP_REASONS = [
    ("safety", r"safety|adverse|toxic|tolerab|side effect|risk|\bs?aes?\b|death|died|fatal|\bharm"),
    ("efficacy", (r"efficacy|futil|lack of (clinical )?(benefit|effect|response|activity)"
                  r"|no (clinical |meaningful |obvious )?(benefit|advantage|activity|response)"
                  r"|(did not|failed to) (meet|show|demonstrate)|endpoints? (was |were )?not met"
                  r"|not effective|ineffective|negative (result|data|outcome)|success criteria")),
    ("enrollment", (r"enrol|accru|recruit|slow|low number|insufficient (number|patients|subjects)"
                    r"|few (patients|subjects|participants)|no (patients|subjects|participants)"
                    r"|lack of (patients|subjects|participants|eligible)|unable to (recruit|identify|enrol)"
                    r"|not enough (patients|subjects|partic)|low rate of")),
    ("business", (r"sponsor|business|strateg|fund|financ|budget|resource|company|portfolio|commercial"
                  r"|priorit|merger|acqui|contract"
                  r"|development (plan|program)|discontinu\w* (the )?development|manufactur|drug supply"
                  r"|supply of|no further need")),
]

# Free-text reasons often negate the very word that would otherwise classify them, e.g.
# "Business decision; no safety concerns". Strip a negated clause before matching so the
# surviving text reflects the actual reason, not the ruled-out one.
NEGATED = (r"\b(not|no|nor|non|unrelated|without|independent of|irrespective of|regardless of|n't)\b[^.;]{0,60}?"
           r"(safety|efficacy|tolerab|toxic|adverse|futil)[^.;]*"
           r"|\b(favou?rable|acceptable|good|well[- ]tolerated|no new)\b[^.;]{0,20}(safety|tolerab)[^.;]*")


def stop_reason(col: Column) -> Column:
    raw = F.lower(F.coalesce(col, F.lit("")))
    text = F.regexp_replace(raw, NEGATED, " ")
    expr = F.lit("other")
    for name, pattern in reversed(STOP_REASONS):
        expr = F.when(text.rlike(pattern), name).otherwise(expr)
    return F.when(F.trim(raw) == "", None).otherwise(expr)


def read_table(spark: SparkSession, aact_dir: str, name: str) -> DataFrame:
    return spark.read.csv(f"{aact_dir}/{name}.txt", sep="|", header=True,
                          multiLine=True, quote='"', escape='"')


def norm(col: Column) -> Column:
    """'Active, not recruiting' -> 'ACTIVE_NOT_RECRUITING'; 'Phase 1/Phase 2' -> 'PHASE1/PHASE2'."""
    c = F.regexp_replace(F.upper(F.trim(col)), r"PHASE\s+", "PHASE")
    return F.regexp_replace(c, r"[^A-Z0-9/]+", "_")


def design_value(col: Column) -> Column:
    """Registry enum, whichever era wrote it: 'Parallel Assignment' and 'PARALLEL' -> 'PARALLEL',
    'N/A' -> 'NA', 'Sponsor-Investigator' -> 'SPONSOR_INVESTIGATOR'. Snapshots before the 2023
    registry modernization use the display wording; current ones use the enum."""
    c = F.regexp_replace(F.regexp_replace(norm(col), r"_+$", ""), r"_ASSIGNMENT$", "")
    return F.when(c == "N/A", "NA").otherwise(c)


def masking_value(col: Column) -> Column:
    """'Double Blind (Subject, Investigator)' -> 'DOUBLE'; 'None (Open Label)' -> 'NONE'."""
    c = norm(col)
    level = F.regexp_extract(c, r"^(NONE|SINGLE|DOUBLE|TRIPLE|QUADRUPLE)", 1)
    return (F.when(c.rlike(r"^(NONE|OPEN_LABEL)"), "NONE")
            .when(level != "", level)
            .otherwise(F.regexp_replace(c, r"_+$", "")))


def flag(col: Column) -> Column:
    """'t'/'true'/'Yes'/'Accepts Healthy Volunteers' -> True, 'f'/'false'/'No' -> False, else null."""
    c = F.lower(F.trim(col))
    return (F.when(c.isin("t", "true", "yes", "y", "1") | c.startswith("accepts"), True)
            .when(c.isin("f", "false", "no", "n", "0"), False))


def study_fields(studies: DataFrame, designs: DataFrame, sponsors: DataFrame) -> DataFrame:
    """Every study's registry-record fields the model uses, unfiltered. build_trials narrows this to
    the eligible population; the point-in-time audit reads the same fields from archived snapshots."""
    agency = norm(F.col("agency_class"))
    lead_sponsor = (sponsors
                    .where(F.lower("lead_or_collaborator") == "lead")
                    .select("nct_id",
                            F.col("name").alias("sponsor_name"),
                            F.when(agency == "INDUSTRY", "INDUSTRY")
                             .when(agency.isin(GOVERNMENT), "GOVERNMENT")
                             .otherwise("OTHER").alias("sponsor_class"))
                    .dropDuplicates(["nct_id"]))

    design = designs.select("nct_id", *[(masking_value if c == "masking" else design_value)(F.col(c)).alias(c)
                                        for c in DESIGN_COLS])
    start_type = F.col("start_date_type") if "start_date_type" in studies.columns else F.lit(None)
    s = studies.select(
        "nct_id",
        norm(F.col("study_type")).alias("study_type"),
        norm(F.col("overall_status")).alias("status"),
        norm(F.col("phase")).alias("phase"),
        F.to_date("start_date").alias("start_date"),
        norm(start_type.cast("string")).alias("start_date_type"),
        F.col("number_of_arms").cast("int").alias("number_of_arms"),
        F.col("brief_title"),                               # display only, never a model input
        F.col("why_stopped"),
        flag(F.col("has_dmc")).alias("has_dmc"),
    )
    return (s.join(design, "nct_id", "left")
            .join(lead_sponsor, "nct_id", "left"))


def build_trials(studies: DataFrame, designs: DataFrame, sponsors: DataFrame,
                 interventions: DataFrame) -> DataFrame:
    drug_trials = (interventions
                   .where(norm(F.col("intervention_type")).isin(DRUG_TYPES))
                   .select("nct_id").distinct())
    s = study_fields(studies, designs, sponsors)
    finished = (F.col("status").isin("COMPLETED", "TERMINATED")
                & (F.col("start_date") >= TRAIN_START))
    eligible = (F.col("study_type") == "INTERVENTIONAL") & F.col("phase").isin(PHASES) \
        & (finished | F.col("status").isin(ACTIVE))

    return (s.where(eligible)
            .join(drug_trials, "nct_id")
            .withColumn("label", F.when(F.col("status") == "TERMINATED", 1)
                                  .when(F.col("status") == "COMPLETED", 0).cast("int"))
            .withColumn("stop_reason", F.when(F.col("label") == 1, stop_reason(F.col("why_stopped"))))
            .drop("study_type", "why_stopped", "start_date_type"))


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
                    F.to_date("start_date").alias("start_date"),
                    F.to_date("completion_date").alias("completion_date"))
            .join(lead, "nct_id"))


def build_sponsor_starts(studies: DataFrame, sponsors: DataFrame) -> DataFrame:
    """Every interventional study's lead sponsor and start date, whatever its current status.

    Unlike sponsor_outcomes (which needs a finished trial to know if it terminated), this
    only needs a start date, so it also covers active trials. Withdrawn trials are excluded:
    they were pulled before enrolling, so they were never really "concurrent" with anything.
    """
    status = norm(F.col("overall_status"))
    lead = (sponsors.where(F.lower("lead_or_collaborator") == "lead")
            .select("nct_id", F.col("name").alias("sponsor_name"))
            .dropDuplicates(["nct_id"]))
    return (studies
            .where((norm(F.col("study_type")) == "INTERVENTIONAL")
                   & (status != "WITHDRAWN")
                   & F.col("start_date").isNotNull())
            .select("nct_id", F.to_date("start_date").alias("start_date"))
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
    build_sponsor_starts(t["studies"], t["sponsors"]) \
        .write.mode("overwrite").parquet(cfg.path("parquet", "sponsor_starts"))
    print(summary)
