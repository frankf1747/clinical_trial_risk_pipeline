"""Run the real feature SQL on DuckDB with hand-built rows, including leakage traps."""
import duckdb
import pytest

from ctrisk.warehouse.sql import SQL_DIR, failed_checks, run_files

FEATURE_FILES = sorted((SQL_DIR / "20_features").glob("*.sql"))


@pytest.fixture(scope="module")
def db():
    con = duckdb.connect()
    run_files(con, [SQL_DIR / "10_raw_tables.sql"])
    con.execute("""INSERT INTO RAW_TRIALS (nct_id, start_date, label, sponsor_name, sponsor_class) VALUES
        ('T1', DATE '2015-06-15', 1,    'Acme',  'INDUSTRY'),
        ('T2', DATE '2019-03-10', 0,    'Acme',  'INDUSTRY'),
        ('T3', DATE '2024-01-01', NULL, 'Beta',  'OTHER'),     -- active, no matched drug
        ('T4', DATE '2010-01-15', 0,    'Gamma', 'OTHER'),     -- its drug is only reported later
        ('T5', DATE '2020-06-01', 1,    'Acme',  'INDUSTRY'),  -- has its own outcome row
        ('T6', DATE '2017-01-01', 0,    'Delta', 'OTHER')      -- first day of the test period""")
    con.execute("""INSERT INTO RAW_TRIAL_ATTRIBUTES (nct_id, n_countries) VALUES
        ('T1', 3), ('T2', 1), ('T3', 1), ('T4', 1), ('T5', 2), ('T6', 1)""")
    con.execute("INSERT INTO RAW_TRIAL_DRUG_MAP VALUES ('T1', 'drugx'), ('T2', 'drugx'), ('T4', 'drugx')")
    # T1 starts 2015-06-15: counted months are < 2015-06; its 12-month window is [2014-06, 2015-06)
    con.execute("""INSERT INTO RAW_FAERS_DRUG_EVENTS (safetyreportid, substance, receivedate, serious, death) VALUES
        ('r4', 'drugx', DATE '2014-01-10', false, true),   -- counted, outside the 12-month window
        ('r5', 'drugx', DATE '2014-05-20', false, false),  -- counted, one month before the window
        ('r6', 'drugx', DATE '2014-06-10', true,  false),  -- counted, first month of the window
        ('r1', 'drugx', DATE '2015-05-20', true,  false),  -- counted, last month of the window
        ('r2', 'drugx', DATE '2015-06-01', false, false),  -- T1's start month: excluded (conservative)
        ('r3', 'drugx', DATE '2015-06-16', true,  true)    -- day after T1 starts: must not count""")
    con.execute("""INSERT INTO RAW_SPONSOR_OUTCOMES VALUES
        ('S1', 1, DATE '2014-12-31', 'Acme'),   -- ended before T1
        ('S3', 1, DATE '2015-06-15', 'Acme'),   -- ended ON T1's start date: not before it
        ('S2', 0, DATE '2015-06-16', 'Acme'),   -- ended the day after T1 started
        ('T1', 1, DATE '2016-01-01', 'Acme'),
        ('T5', 1, DATE '2020-05-01', 'Acme')    -- T5's own row: never its own history""")
    run_files(con, FEATURE_FILES)
    return con


def features(db, nct_id):
    cur = db.execute("SELECT * FROM TRIAL_FEATURES WHERE nct_id = ?", [nct_id])
    return dict(zip([c[0] for c in cur.description], cur.fetchone()))


def test_faers_counts_only_reports_before_the_start_month(db):
    t1 = features(db, "T1")
    assert t1["faers_reports"] == 4               # r4, r5, r6, r1 — not r2 (start month) or r3 (after)
    assert t1["faers_reports_12m"] == 2           # r6, r1 — r5 is one month too early
    assert t1["faers_serious_share"] == 0.5       # r6, r1 of 4
    assert t1["faers_death_share"] == 0.25        # r4 of 4
    assert t1["has_faers_history"] is True
    assert features(db, "T2")["faers_reports"] == 6


def test_drug_first_reported_after_start_does_not_count(db):
    # Counting it would reveal that the drug later reached the market
    t4 = features(db, "T4")
    assert (t4["n_substances"], t4["faers_reports"], t4["has_faers_history"]) == (0, 0, False)


def test_trial_without_matched_drugs_has_no_history(db):
    t3 = features(db, "T3")
    assert (t3["n_substances"], t3["faers_reports"], t3["has_faers_history"]) == (0, 0, False)
    assert t3["faers_serious_share"] is None


def test_sponsor_history_uses_only_trials_that_ended_before_start(db):
    t1 = features(db, "T1")
    assert (t1["sponsor_prior_trials"], t1["sponsor_prior_termination_rate"]) == (1, 1.0)  # S1 only
    t2 = features(db, "T2")
    assert t2["sponsor_prior_trials"] == 4                                    # S1, S3, S2, T1
    assert t2["sponsor_prior_termination_rate"] == pytest.approx(3 / 4)
    t5 = features(db, "T5")
    assert t5["sponsor_prior_trials"] == 4                                    # same four, not T5 itself
    assert features(db, "T3")["sponsor_prior_trials"] == 0


def test_split_by_start_year(db):
    splits = {t: features(db, t)["split"] for t in ("T1", "T2", "T3", "T4", "T5", "T6")}
    assert splits == {"T1": "train", "T2": "test", "T3": "score",
                      "T4": "train", "T5": "test", "T6": "test"}


def test_post_start_fields_never_reach_features(db):
    cols = {c[0] for c in db.execute("SELECT * FROM TRIAL_FEATURES LIMIT 0").description}
    assert not cols & {"status", "completion_date", "sponsor_name"}


def test_checks_pass(db):
    assert failed_checks(db, SQL_DIR / "90_checks.sql") == []
