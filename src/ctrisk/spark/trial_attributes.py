"""Per-trial attributes from the registration record: geography, eligibility, disease area, who runs it, what it measures.

Countries include ones later removed from the record, so the count is every country the
trial ever listed. Record edits can still shift it; M4 measures the effect.
"""
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from ctrisk.config import load_config
from ctrisk.spark.clean_trials import read_table
from ctrisk.spark.session import get_spark

# Top-level MeSH disease categories, as AACT lists them among each trial's condition ancestors
AREAS = {
    "neoplasms": "Neoplasms",
    "cardiovascular": "Cardiovascular Diseases",
    "nervous_system": "Nervous System Diseases",
    "mental": "Mental Disorders",
    "infections": "Infections",
    "respiratory": "Respiratory Tract Diseases",
    "digestive": "Digestive System Diseases",
    "metabolic": "Nutritional and Metabolic Diseases",
    "immune": "Immune System Diseases",
    "skin": "Skin and Connective Tissue Diseases",
    "musculoskeletal": "Musculoskeletal Diseases",
    "urogenital": "Urogenital Diseases",
    "blood": "Hemic and Lymphatic Diseases",
    "endocrine": "Endocrine System Diseases",
}


def age_years(col: Column) -> Column:
    """'18 Years' -> 18.0, '6 Months' -> 0.5; anything else -> null."""
    n = F.regexp_extract(col, r"(\d+(?:\.\d+)?)", 1).cast("double")
    unit = F.lower(F.regexp_extract(col, r"(?i)(year|month|week|day)", 1))
    per_year = F.when(unit == "year", 1).when(unit == "month", 12).when(unit == "week", 52) \
        .when(unit == "day", 365)
    return F.round(n / per_year, 2)


def _is_criterion(x: Column) -> Column:
    return (F.trim(x) != "") & ~F.lower(F.trim(x)).rlike(r"criteria\s*:?$")


def build_trial_attributes(trials: DataFrame, countries: DataFrame, eligibilities: DataFrame,
                           browse_conditions: DataFrame, sponsors: DataFrame,
                           responsible_parties: DataFrame, keywords: DataFrame) -> DataFrame:
    geo = countries.groupBy("nct_id").agg(
        F.countDistinct("name").alias("n_countries"),
        F.expr("bool_and(name = 'United States')").alias("us_only"))

    # AACT stores line breaks in criteria as '~'; each non-empty, non-header line is one criterion
    lines = F.split(F.coalesce(F.col("criteria"), F.lit("")), "~")
    elig = eligibilities.select(
        "nct_id",
        age_years(F.col("minimum_age")).alias("min_age_years"),
        age_years(F.col("maximum_age")).alias("max_age_years"),
        (F.col("healthy_volunteers") == "t").alias("healthy_volunteers"),
        F.upper("gender").alias("sex"),
        F.size(F.filter(lines, _is_criterion)).alias("criteria_count"),
        F.length(F.coalesce(F.col("criteria"), F.lit(""))).alias("criteria_chars"))

    area = (browse_conditions
            .where(F.col("mesh_term").isin(list(AREAS.values())))
            .groupBy("nct_id")
            .agg(*[F.max(F.col("mesh_term") == term).alias(f"area_{key}") for key, term in AREAS.items()]))

    party = (responsible_parties
             .select("nct_id", F.upper("responsible_party_type").alias("responsible_party"))
             .dropDuplicates(["nct_id"]))
    collab = sponsors.groupBy("nct_id").agg(
        F.sum((F.lower("lead_or_collaborator") == "collaborator").cast("int")).alias("n_collaborators"))
    kw = keywords.groupBy("nct_id").agg(F.count("*").alias("n_keywords"))

    counts = ["n_countries", "n_collaborators", "n_keywords"]
    return (trials.select("nct_id")
            .join(geo, "nct_id", "left")
            .join(elig, "nct_id", "left")
            .join(area, "nct_id", "left")
            .join(party, "nct_id", "left")
            .join(collab, "nct_id", "left")
            .join(kw, "nct_id", "left")
            .fillna(0, subset=counts)
            .fillna(False, subset=["us_only", *[f"area_{k}" for k in AREAS]]))


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("trial_attributes", cfg.mode)
    aact = cfg.path("raw", "aact")
    attrs = build_trial_attributes(spark.read.parquet(cfg.path("parquet", "trials")),
                                   *[read_table(spark, aact, t) for t in
                                     ("countries", "eligibilities", "browse_conditions", "sponsors",
                                      "responsible_parties", "keywords")])
    attrs.write.mode("overwrite").parquet(cfg.path("parquet", "trial_attributes"))
    print({"trials": attrs.count()})
