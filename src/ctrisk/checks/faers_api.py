"""Compare our FAERS report counts with the openFDA API for a few substances in one quarter.

openFDA's bulk quarter folders are split by receiptdate, not receivedate, so this check filters
and searches by receiptdate to match the file we actually downloaded.

openFDA's search matches words inside substance names, so the API can count slightly more
(e.g. salt variants). A ratio near 1.0 means ingest and flatten lost nothing.
"""
import requests
from pyspark.sql import functions as F

from ctrisk.config import load_config
from ctrisk.spark.session import get_spark

API = "https://api.fda.gov/drug/event.json"
SUBSTANCES = ["ondansetron", "metformin", "atorvastatin", "adalimumab"]
START, END = "2016-01-01", "2016-03-31"


def api_count(substance: str) -> int:
    search = (f"receiptdate:[{START.replace('-', '')} TO {END.replace('-', '')}]"
              f' AND patient.drug.openfda.substance_name:"{substance}"')
    r = requests.get(API, params={"search": search, "limit": 1}, timeout=60)
    if r.status_code == 404:  # openFDA answers 404 when nothing matches
        return 0
    r.raise_for_status()
    return r.json()["meta"]["results"]["total"]


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("check_faers", cfg.mode)
    ours = dict(spark.read.parquet(cfg.path("parquet", "faers_drug_events"))
                .where(F.col("harmonized") & F.col("receiptdate").between(START, END)
                       & F.col("substance").isin(SUBSTANCES))
                .groupBy("substance").agg(F.countDistinct("safetyreportid"))
                .collect())
    print(f"{'substance':<14}{'ours':>8}{'api':>8}{'ratio':>8}")
    for s in SUBSTANCES:
        api = api_count(s)
        print(f"{s:<14}{ours.get(s, 0):>8}{api:>8}{ours.get(s, 0) / api if api else 0:>8.2f}")
