"""FAERS bulk zips -> one row per (report, substance), latest report version only.

Where each substance name came from:
  harmonized        openFDA's standard name (openfda.substance_name)
  active_substance  FDA's coded active ingredient, used when openFDA has no standard name
  neither           free-text product name as reported; kept for counting, never for matching

Each openFDA file is a single JSON document ({"meta":..., "results":[...]}), so each Spark task
streams one zip with ijson instead of loading it whole.
"""
import io
import os
import zipfile
from collections.abc import Iterator

import ijson
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ctrisk.config import load_config
from ctrisk.drugnames import normalize
from ctrisk.spark.session import get_spark

SCHEMA = ("safetyreportid string, version int, receivedate string, receiptdate string, "
          "serious boolean, death boolean, suspect boolean, harmonized boolean, active_substance boolean, "
          "substance string")


def report_rows(report: dict) -> Iterator[tuple]:
    rid = report.get("safetyreportid")
    if not rid:
        return
    v = report.get("safetyreportversion")
    version = int(v) if str(v).isdigit() else 1
    base = (rid, version, report.get("receivedate"), report.get("receiptdate"),
            report.get("serious") == "1", report.get("seriousnessdeath") == "1")
    for drug in (report.get("patient") or {}).get("drug") or []:
        harmonized = drug.get("openfda", {}).get("substance_name")
        active = (drug.get("activesubstance") or {}).get("activesubstancename")
        names = harmonized or [active or drug.get("medicinalproduct") or ""]
        for substance in sorted({normalize(n) for n in names} - {""}):
            yield (*base, drug.get("drugcharacterization") == "1",
                   bool(harmonized), not harmonized and bool(active), substance)


def parse_zip(content: bytes) -> Iterator[tuple]:
    with zipfile.ZipFile(io.BytesIO(content)) as z, z.open(z.namelist()[0]) as f:
        for report in ijson.items(f, "results.item"):
            yield from report_rows(report)


def build_drug_events(spark: SparkSession, zip_glob: str) -> DataFrame:
    rows = spark.sparkContext.binaryFiles(zip_glob).flatMap(lambda kv: parse_zip(kv[1]))
    latest = F.max("version").over(Window.partitionBy("safetyreportid"))
    return (spark.createDataFrame(rows, SCHEMA)
            .withColumn("latest", latest)  # window functions can't go directly in where()
            .where(F.col("version") == F.col("latest"))
            .groupBy("safetyreportid", "substance")
            .agg(F.max("receivedate").alias("receivedate"), F.max("receiptdate").alias("receiptdate"),
                 F.max("serious").alias("serious"), F.max("death").alias("death"),
                 F.max("suspect").alias("suspect"), F.max("harmonized").alias("harmonized"),
                 F.max("active_substance").alias("active_substance"))
            .withColumn("receivedate", F.to_date("receivedate", "yyyyMMdd"))
            .withColumn("receiptdate", F.to_date("receiptdate", "yyyyMMdd")))


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("flatten_faers", cfg.mode)
    out = cfg.path("parquet", "faers_drug_events")
    raw = os.getenv("FAERS_RAW", "raw/faers")              # raw/faers_full: the whole history, ingested in the cloud
    build_drug_events(spark, cfg.path(*raw.split("/"), "*", "*.zip")).write.mode("overwrite").parquet(out)
    events = spark.read.parquet(out)
    print({"rows": events.count(),
           "reports": events.select("safetyreportid").distinct().count(),
           "substances": events.select("substance").distinct().count()})
