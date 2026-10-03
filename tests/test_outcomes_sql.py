import duckdb

from ctrisk.warehouse.sql import SQL_DIR, run_files


def test_outcomes_carry_end_dates_for_finished_trials_only():
    con = duckdb.connect()
    con.execute("CREATE TABLE TRIAL_FEATURES (nct_id STRING, split STRING, start_date DATE, label INT)")
    con.execute("""INSERT INTO TRIAL_FEATURES VALUES ('T', 'train', DATE '2010-01-01', 1),
        ('C', 'test', DATE '2015-01-01', 0), ('A', 'score', DATE '2024-01-01', NULL)""")
    con.execute("CREATE TABLE RAW_SPONSOR_OUTCOMES (nct_id STRING, terminated INT, start_date DATE, completion_date DATE)")
    con.execute("""INSERT INTO RAW_SPONSOR_OUTCOMES VALUES ('T', 1, DATE '2010-01-01', DATE '2011-06-30'),
        ('C', 0, DATE '2015-01-01', DATE '2017-01-01'), ('A', 0, DATE '2024-01-01', DATE '2030-01-01')""")
    run_files(con, [SQL_DIR / "50_outcomes.sql"])
    rows = dict(con.execute("SELECT nct_id, CAST(end_date AS STRING) FROM TRIAL_OUTCOMES").fetchall())
    assert rows == {"T": "2011-06-30", "C": "2017-01-01", "A": None}     # a running trial has no end yet
