from pathlib import Path

import pytest
from pyspark.sql import SparkSession

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder.master("local[1]")
         .config("spark.sql.shuffle.partitions", "1")
         .config("spark.sql.session.timeZone", "UTC")
         .getOrCreate())
    yield s
    s.stop()


@pytest.fixture(scope="session")
def aact(spark):
    from ctrisk.spark.clean_trials import read_table
    return {t: read_table(spark, str(FIXTURES / "aact"), t)
            for t in ("studies", "designs", "sponsors", "interventions")}
