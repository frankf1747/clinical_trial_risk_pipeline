"""The lookup view, run on DuckDB with hand-built rows."""
import duckdb
import pytest

from ctrisk.warehouse.sql import SQL_DIR, run_files


@pytest.fixture(scope="module")
def db():
    con = duckdb.connect()
    con.execute("""CREATE TABLE RAW_TRIALS (nct_id STRING, brief_title STRING, phase STRING, sponsor_name STRING,
        sponsor_class STRING, status STRING, start_date DATE)""")
    con.execute("""INSERT INTO RAW_TRIALS VALUES
        ('A', 'Drug A in lung cancer', 'PHASE2', 'Acme', 'INDUSTRY', 'RECRUITING', DATE '2024-01-01'),
        ('B', 'Drug B in asthma',      'PHASE3', 'State U', 'OTHER', 'COMPLETED', DATE '2010-05-01')""")
    con.execute("""CREATE TABLE TRIAL_LOOKUP_SCORES (nct_id STRING, model_version STRING, score_type STRING,
        risk_score FLOAT, risk_percentile FLOAT, enrollment_risk_score FLOAT, reasons STRING)""")
    con.execute("""INSERT INTO TRIAL_LOOKUP_SCORES VALUES
        ('A', 'v4',  'forward',     0.30, 91.0, 0.12, '[]'),
        ('A', 'v10', 'forward',     0.35, 93.0, 0.10, '[]'),     -- v10 is newer than v4, not older
        ('B', 'v4',  'out_of_fold', 0.05, 12.5, 0.01, '[]'),
        ('Z', 'v4',  'forward',     0.20, 50.0, 0.05, '[]')""")  # Z: no registry row, left out
    run_files(con, [SQL_DIR / "40_lookup.sql"])
    return con


def test_lookup_has_one_row_per_trial_from_the_newest_version(db):
    rows = db.execute("SELECT nct_id, model_version, risk_score, sponsor_name, status FROM TRIAL_LOOKUP "
                      "ORDER BY nct_id").fetchall()
    assert rows == [("A", "v10", pytest.approx(0.35), "Acme", "RECRUITING"),
                    ("B", "v4", pytest.approx(0.05), "State U", "COMPLETED")]
