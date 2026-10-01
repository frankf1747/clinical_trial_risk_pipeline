"""Registry-record features for the 2017-2020 cohort as the records stood in archived AACT snapshots.

The pipeline's other registry features come from the latest version of each record. This rebuilds the
same fields (design, eligibility, geography, text) from each archive, keeps the cohort's trials, and
picks per trial the last archive dated on or before its start: what was on the registry when it began.
Trials first seen in an archive after their start (registered late) fall back to the first archive
that has them; lag_days (archive date minus start date) says how far off the record is.

Usage: python -m ctrisk.spark.point_in_time <archive zip or URL> [...]
Each archive is extracted, reduced to cohort rows, and deleted, so disk use stays at one archive.
Archives already reduced are skipped (FORCE=1 redoes them).
"""
import os
import re
import shutil
import sys
from pathlib import Path

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from ctrisk.config import load_config
from ctrisk.ingest.aact import fetch
from ctrisk.spark.clean_trials import read_table, study_fields
from ctrisk.spark.session import get_spark
from ctrisk.spark.trial_attributes import build_trial_attributes
from ctrisk.spark.trial_text import build_trial_text

TABLES = ("studies", "designs", "sponsors", "countries", "eligibilities", "browse_conditions",
          "responsible_parties", "keywords", "brief_summaries")
COHORT = ("2017-01-01", "2021-01-01")          # start dates the monthly AACT archives can cover


def archive_date(name: str) -> str:
    """'20170101_pipe-delimited-export.zip' -> '2017-01-01'."""
    m = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})", Path(name).name)
    if not m:
        raise ValueError(f"{name}: no date in the file name; rename it YYYYMMDD_*.zip")
    return "-".join(m.groups())


# Archived exports (before the 2023 registry modernization) hard-wrap criteria at ~80 characters, with
# '~' or only a run of spaces between wrapped lines; current records keep one criterion per '~' line with
# markdown bullets. Without unwrapping, criteria_count and criteria_chars differ for nearly every trial.
_BREAK = re.compile(r"\s*[~\n]\s*|\s{2,}(?=(?:[-*•o]|\d{1,2}[.)]|[a-z][.)])\s)"
                    r"|\s{2,}(?=(?:inclusion|exclusion)[^:]{0,40}criteria\s*:)", re.IGNORECASE)
_ITEM = re.compile(r"^(?:[-*•o]|\d{1,2}[.)]|[a-z][.)])\s")
_HEADER = re.compile(r"criteria\s*:?$", re.IGNORECASE)


def unwrap_criteria(text: str | None) -> str | None:
    """An archived criteria block in the current record's layout: one line per criterion, '* ' bullets.
    Matches the current snapshot's criterion count for ~3 in 4 trials; the rest include real edits."""
    if text is None:
        return None
    out: list[str] = []
    for seg in _BREAK.split(text):
        line = re.sub(r"\s+", " ", seg).strip()
        if not line:
            continue
        if _ITEM.match(line) or _HEADER.search(line) or not out or _HEADER.search(out[-1]):
            out.append(re.sub(r"^[-•o]\s+", "* ", line))
        else:
            out[-1] += " " + line                                  # a wrapped continuation
    return "~".join(out)


def registry_features(t: dict, cohort: DataFrame, date: str) -> DataFrame:
    """The model's registry-derived columns for cohort trials present in this archive, tagged with its date."""
    fields = (study_fields(t["studies"], t["designs"], t["sponsors"])
              .join(cohort.select("nct_id").distinct(), "nct_id")
              .withColumnRenamed("status", "archive_status")
              .withColumnRenamed("start_date", "archive_start_date")
              .drop("study_type", "why_stopped", "sponsor_name", "brief_title")      # not model inputs
              .cache())
    unwrap = F.udf(unwrap_criteria, StringType())
    t = {**t, "eligibilities": t["eligibilities"].join(cohort.select("nct_id").distinct(), "nct_id", "left_semi")
                                                .withColumn("criteria", unwrap(F.col("criteria")))}
    attrs = build_trial_attributes(fields, t["countries"], t["eligibilities"], t["browse_conditions"],
                                   t["sponsors"], t["responsible_parties"], t["keywords"])
    if "mesh_type" not in t["browse_conditions"].columns:
        # Older archives list only each trial's own MeSH terms, not their ancestors, so the top-level
        # disease areas would read False for nearly everyone. Unknown, not False; the audit keeps the latest.
        attrs = attrs.select(*[F.lit(None).cast("boolean").alias(c) if c.startswith("area_") else F.col(c)
                               for c in attrs.columns])
    text = build_trial_text(fields, t["studies"], t["brief_summaries"], t["eligibilities"], t["keywords"])
    return (fields.join(attrs, "nct_id", "left").join(text, "nct_id", "left")
            .withColumn("archive_date", F.to_date(F.lit(date))))


def nearest_archive(seen: DataFrame, starts: DataFrame) -> DataFrame:
    """One row per trial: its last archived record on or before start, else its first one after."""
    lag = F.datediff("archive_date", "start_date")
    ranked = (seen.join(starts.where(F.col("start_date").isNotNull()), "nct_id")
              .withColumn("lag_days", lag)
              .withColumn("_rank", F.row_number().over(Window.partitionBy("nct_id").orderBy(
                  (F.col("lag_days") > 0).asc(),                           # on/before start first
                  F.when(F.col("lag_days") <= 0, -F.col("lag_days")).otherwise(F.col("lag_days")).asc()))))
    return ranked.where(F.col("_rank") == 1).drop("_rank", "start_date")


def reduce_archive(spark, cfg, source: str, cohort: DataFrame) -> str:
    date = archive_date(source)
    out = cfg.path("parquet_pit", "by_archive", date)
    if Path(out, "_SUCCESS").exists() and os.getenv("FORCE") != "1":
        return f"{date}: already reduced"
    folder = Path(cfg.local_root, "raw", "aact_archive", date)
    fetch(source, folder, TABLES)
    t = {name: read_table(spark, str(folder), name) for name in TABLES}
    registry_features(t, cohort, date).write.mode("overwrite").parquet(out)
    shutil.rmtree(folder)
    if source.startswith(("http://", "https://")):
        folder.with_name(f"{date}.zip").unlink(missing_ok=True)
    return f"{date}: {spark.read.parquet(out).count()} cohort trials"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cfg = load_config()
    spark = get_spark("point_in_time", cfg.mode)
    trials = spark.read.parquet(cfg.path("parquet", "trials"))
    cohort = trials.where(F.col("label").isNotNull() & (F.col("start_date") >= COHORT[0])
                          & (F.col("start_date") < COHORT[1])).select("nct_id", "start_date").cache()
    for source in sys.argv[1:]:
        print(reduce_archive(spark, cfg, source, cohort))

    seen = spark.read.parquet(cfg.path("parquet_pit", "by_archive", "*"))
    picked = nearest_archive(seen, cohort).cache()
    picked.write.mode("overwrite").parquet(cfg.path("parquet_pit", "registry_at_start"))

    # Format drift shows up as a column that is empty (or, for filled flags, never true) in some archives
    # only; a step change between neighbouring archives is a format change, not record edits. Read this
    # before the audit.
    print("by archive: share of cohort rows with a null value; for area_neoplasms, the share flagged:")
    cols = ["phase", "masking", "allocation", "healthy_volunteers", "responsible_party", "criteria_chars", "text"]
    seen.groupBy("archive_date").agg(
        F.count("*").alias("trials"),
        *[F.round(F.avg(F.col(c).isNull().cast("int")), 3).alias(c) for c in cols],
        F.round(F.avg(F.col("area_neoplasms").cast("int")), 3).alias("area_neoplasms"),
    ).orderBy("archive_date").show(100)
    lag = picked.agg(F.count("*"), F.expr("percentile(lag_days, 0.5)"), F.sum((F.col("lag_days") > 0).cast("int")))
    total, median, late = lag.first()
    print({"cohort": cohort.count(), "with_archived_record": total, "median_lag_days": median,
           "first_seen_after_start": late})
