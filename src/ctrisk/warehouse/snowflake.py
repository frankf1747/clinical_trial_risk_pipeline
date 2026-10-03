"""Load Parquet from the GCS stage into Snowflake, build TRIAL_FEATURES, and check it."""
import os

import snowflake.connector

from ctrisk.config import load_config
from ctrisk.gates import DataGateError
from ctrisk.warehouse.sql import SQL_DIR, failed_checks, run_files

BUILD = [SQL_DIR / "10_raw_tables.sql", SQL_DIR / "11_copy.sql",
         *sorted((SQL_DIR / "20_features").glob("*.sql")),
         SQL_DIR / "50_outcomes.sql"]                       # durations for the survival model, apart from features


def connect():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        private_key_file=os.path.expanduser(os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"]),
        role=os.getenv("SNOWFLAKE_ROLE", "SYSADMIN"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "CTRISK_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "CTRISK"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PIPELINE"),
    )


if __name__ == "__main__":
    cfg = load_config()  # also loads .env
    with connect() as conn, conn.cursor() as cur:
        cur.execute("LIST @PARQUET_STAGE/trials/")
        if not cur.fetchall():
            raise DataGateError("stage has no trials Parquet; run `make upload` first")
        run_files(cur, BUILD)
        if failures := failed_checks(cur, SQL_DIR / "90_checks.sql"):
            raise DataGateError("; ".join(failures))
        labeled = cur.execute("SELECT COUNT(label) FROM TRIAL_FEATURES").fetchone()[0]
        if labeled < cfg.min_trials:
            raise DataGateError(f"{labeled} labeled trials in TRIAL_FEATURES, expected {cfg.min_trials}+")
        cur.execute("""SELECT split, COUNT(*), ROUND(AVG(label), 3), ROUND(AVG(has_faers_history::INT), 3)
                       FROM TRIAL_FEATURES GROUP BY split ORDER BY split""")
        print("split | trials | termination rate | share with FAERS history")
        for row in cur.fetchall():
            print(" | ".join(str(v) for v in row))
