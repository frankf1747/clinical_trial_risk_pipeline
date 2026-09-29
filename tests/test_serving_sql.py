"""The reviewer-facing views, run on DuckDB with hand-built rows."""
import duckdb
import pytest

from ctrisk.spark.trial_attributes import AREAS
from ctrisk.warehouse.sql import SQL_DIR, run_files


@pytest.fixture(scope="module")
def db():
    con = duckdb.connect()
    con.execute("CREATE TABLE RAW_TRIALS (nct_id STRING, brief_title STRING, phase STRING, sponsor_class STRING)")
    con.execute("""INSERT INTO RAW_TRIALS VALUES
        ('A', 'Drug A in lung cancer', 'PHASE2', 'INDUSTRY'),
        ('B', 'Drug B in asthma',      'PHASE3', 'OTHER'),
        ('C', 'Drug C, finished',      'PHASE1', 'OTHER')""")
    areas = ", ".join(f"area_{a} BOOLEAN DEFAULT false" for a in AREAS)
    con.execute(f"CREATE TABLE TRIAL_FEATURES (nct_id STRING, split STRING, start_date DATE, {areas})")
    con.execute("""INSERT INTO TRIAL_FEATURES (nct_id, split, start_date, area_neoplasms, area_respiratory, area_immune)
        VALUES ('A', 'score', DATE '2024-01-01', true,  true,  false),   -- both: oncology wins
               ('B', 'score', DATE '2023-05-01', false, true,  false),
               ('C', 'test',  DATE '2015-02-01', false, false, true)""")
    con.execute("""CREATE TABLE TRIAL_RISK_SCORES (nct_id STRING, model_version STRING, risk_score FLOAT,
        risk_decile INT, enrollment_risk_score FLOAT, top_driver_1 STRING, top_driver_2 STRING,
        top_driver_3 STRING, scored_at STRING)""")
    con.execute("""INSERT INTO TRIAL_RISK_SCORES VALUES
        ('A', 'v1', 0.10, 5,  NULL, 'phase', NULL, NULL, '2026-09-27 10:00:00'),
        ('A', 'v2', 0.30, 10, 0.12, 'registration_text', 'us_only', NULL, '2026-09-28 10:00:00'),
        ('B', 'v2', 0.05, 1,  0.01, 'phase', NULL, NULL, '2026-09-28 10:00:00'),
        ('C', 'v2', 0.50, 10, 0.20, 'phase', NULL, NULL, '2026-09-28 10:00:00')""")
    run_files(con, [SQL_DIR / "30_serving.sql"])
    return con


def rows(db, sql):
    cur = db.execute(sql)
    return [dict(zip([c[0].lower() for c in cur.description], r)) for r in cur.fetchall()]


def test_active_view_keeps_the_latest_score_of_active_trials_only(db):
    got = {r["nct_id"]: r for r in rows(db, "SELECT * FROM VW_ACTIVE_TRIAL_RISK")}
    assert set(got) == {"A", "B"}                                  # C is a finished trial
    assert (got["A"]["model_version"], got["A"]["risk_decile"]) == ("v2", 10)
    assert got["A"]["brief_title"] == "Drug A in lung cancer"


def test_one_disease_area_per_trial_in_a_fixed_order(db):
    got = {r["nct_id"]: r["disease_area"] for r in rows(db, "SELECT * FROM VW_ACTIVE_TRIAL_RISK")}
    assert got == {"A": "Oncology", "B": "Respiratory"}


def test_area_summary(db):
    got = {r["disease_area"]: r for r in rows(db, "SELECT * FROM VW_RISK_BY_AREA")}
    assert got["Oncology"]["trials"] == 1 and got["Oncology"]["top_decile_trials"] == 1
    assert got["Respiratory"]["avg_risk"] == pytest.approx(0.05)
