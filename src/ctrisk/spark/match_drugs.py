"""Map each trial to the FAERS substances it tests -> trial_drug_map(nct_id, substance)."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ctrisk.config import load_config
from ctrisk.drugnames import is_specific, match_substances, normalize
from ctrisk.gates import DataGateError
from ctrisk.spark.clean_trials import MODEL_END, read_table
from ctrisk.spark.session import get_spark


def trial_drug_names(drug_interventions: DataFrame, other_names: DataFrame,
                     browse_interventions: DataFrame) -> DataFrame:
    """Every name a trial uses for its drugs: intervention names, other names, MeSH terms."""
    trials = drug_interventions.select("nct_id").distinct()
    others = (other_names
              .join(drug_interventions.select("intervention_id"), "intervention_id", "left_semi")
              .select("nct_id", "name"))
    mesh = (browse_interventions
            .where(F.col("mesh_type") == "mesh-list")
            .join(trials, "nct_id", "left_semi")
            .select("nct_id", F.col("mesh_term").alias("name")))
    return drug_interventions.select("nct_id", "name").unionByName(others).unionByName(mesh).distinct()


def faers_vocab(drug_events: DataFrame) -> frozenset[str]:
    """Substances with an FDA-coded name, minus catch-alls like 'vitamin d nos'."""
    coded = drug_events.where(F.col("harmonized") | F.col("active_substance"))
    rows = coded.select("substance").distinct().collect()
    return frozenset(r.substance for r in rows if is_specific(r.substance))


def build_trial_drug_map(names: DataFrame, vocab: frozenset[str]) -> DataFrame:
    shared = names.sparkSession.sparkContext.broadcast(vocab)

    @F.udf(T.ArrayType(T.StringType()))
    def substances(name):
        return match_substances(normalize(name or ""), shared.value)

    return names.select("nct_id", F.explode(substances("name")).alias("substance")).distinct()


def check_match_rate(trials: DataFrame, trial_drug_map: DataFrame, min_rate: float) -> dict:
    """Share of the modelled trials (labeled, started before MODEL_END) with at least one matched drug."""
    labeled = trials.where(F.col("label").isNotNull() & (F.col("start_date") < MODEL_END)).select("nct_id")
    total = labeled.count()
    matched = labeled.join(trial_drug_map, "nct_id", "left_semi").count()
    rate = matched / total if total else 0.0
    if rate < min_rate:
        raise DataGateError(f"drug match rate {rate:.1%} is below the {min_rate:.0%} minimum")
    return {"labeled_trials": total, "matched": matched, "match_rate": round(rate, 3),
            "substances": trial_drug_map.select("substance").distinct().count()}


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("match_drugs", cfg.mode)
    aact = cfg.path("raw", "aact")
    names = trial_drug_names(spark.read.parquet(cfg.path("parquet", "drug_interventions")),
                             read_table(spark, aact, "intervention_other_names"),
                             read_table(spark, aact, "browse_interventions"))
    vocab = faers_vocab(spark.read.parquet(cfg.path("parquet", "faers_drug_events")))
    drug_map = build_trial_drug_map(names, vocab).cache()
    summary = check_match_rate(spark.read.parquet(cfg.path("parquet", "trials")), drug_map,
                               cfg.min_match_rate)
    drug_map.write.mode("overwrite").parquet(cfg.path("parquet", "trial_drug_map"))
    print({"vocab": len(vocab), **summary})
