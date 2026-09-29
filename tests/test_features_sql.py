"""Run the real feature SQL on DuckDB with hand-built rows, including leakage traps."""
import duckdb
import pytest

from ctrisk.warehouse.sql import SQL_DIR, failed_checks, run_files

FEATURE_FILES = sorted((SQL_DIR / "20_features").glob("*.sql"))


@pytest.fixture(scope="module")
def db():
    con = duckdb.connect()
    run_files(con, [SQL_DIR / "10_raw_tables.sql"])
    con.execute("""INSERT INTO RAW_TRIALS (nct_id, start_date, label, stop_reason, sponsor_name, sponsor_class) VALUES
        ('T1', DATE '2015-06-15', 1,    'enrollment', 'Acme',  'INDUSTRY'),
        ('T2', DATE '2019-03-10', 0,    NULL,         'Acme',  'INDUSTRY'),
        ('T3', DATE '2024-01-01', NULL, NULL,         'Beta',  'OTHER'),     -- active, no matched drug
        ('T4', DATE '2010-01-15', 0,    NULL,         'Gamma', 'OTHER'),     -- its drug is only reported later
        ('T5', DATE '2020-06-01', 1,    'business',   'Acme',  'INDUSTRY'),  -- has its own outcome row
        ('T6', DATE '2017-01-01', 0,    NULL,         'Delta', 'OTHER'),     -- first day of the recent years
        ('T7', DATE '2015-01-01', 0,    NULL,         'Eps',   'OTHER'),     -- first day of the test years
        ('T8', DATE '2014-12-31', 0,    NULL,         'Eps',   'OTHER'),     -- last day of the training years
        ('T9', DATE '2021-01-01', 0,    NULL,         'Eps',   'OTHER')      -- finished, started after 2020""")
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
    con.execute("INSERT INTO RAW_TRIAL_TEXT VALUES ('T1', 'Phase 3 trial of drugx in adults'), ('T2', 'Pilot study')")
    con.execute("""INSERT INTO RAW_SPONSOR_OUTCOMES VALUES
        ('S1', 1, DATE '2013-01-01', DATE '2014-12-31', 'Acme'),   -- ended before T1
        ('S3', 1, DATE '2014-01-01', DATE '2015-06-15', 'Acme'),   -- ended ON T1's start date: concurrent, not prior
        ('S2', 0, DATE '2015-01-01', DATE '2015-06-16', 'Acme'),   -- ended the day after T1 started: concurrent
        ('T1', 1, DATE '2015-06-15', DATE '2016-01-01', 'Acme'),
        ('T5', 1, DATE '2020-06-01', DATE '2020-05-01', 'Acme')    -- T5's own row: never its own history""")
    # T1 starts 2015-06-15: its 24-month window is [2013-06-15, 2015-06-15).
    con.execute("""INSERT INTO RAW_SPONSOR_STARTS VALUES
        ('X0', 'Acme', DATE '2013-06-14'),   -- one day before the cutoff: outside the window
        ('X1', 'Acme', DATE '2013-07-01'),   -- inside
        ('X2', 'Acme', DATE '2014-01-01'),   -- inside
        ('X3', 'Acme', DATE '2015-06-14'),   -- inside, the day before T1 starts
        ('T1', 'Acme', DATE '2015-06-15')    -- T1's own row: never its own history""")
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
    splits = {t: features(db, t)["split"] for t in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")}
    assert splits == {"T1": "test", "T2": "recent", "T3": "score", "T4": "train",
                      "T5": "recent", "T6": "recent", "T7": "test", "T8": "train"}


def test_post_start_fields_never_reach_features(db):
    cols = {c[0] for c in db.execute("SELECT * FROM TRIAL_FEATURES LIMIT 0").description}
    assert not cols & {"status", "completion_date", "sponsor_name", "stop_reason"}


def test_checks_pass(db):
    assert failed_checks(db, SQL_DIR / "90_checks.sql") == []


def test_text_is_carried_but_missing_text_is_null(db):
    assert features(db, "T1")["text"].startswith("Phase 3")
    assert features(db, "T3")["text"] is None


def test_sponsor_trials_started_2y_counts_starts_in_the_24_months_before(db):
    # T1 starts 2015-06-15: window is [2013-06-15, 2015-06-15).
    # X0 (2013-06-14) is one day before the cutoff: excluded.
    # X1 (2013-07-01), X2 (2014-01-01), X3 (2015-06-14) fall inside: 3.
    # T1's own row (2015-06-15) is both on-or-after its own start and its own nct_id: excluded either way.
    assert features(db, "T1")["sponsor_trials_started_2y"] == 3
    assert features(db, "T2")["sponsor_trials_started_2y"] == 0      # nothing started in T2's own window
    assert features(db, "T3")["sponsor_trials_started_2y"] == 0


def test_reason_targets(db):
    t1, t2, t5, t3 = (features(db, t) for t in ("T1", "T2", "T5", "T3"))
    assert (t1["label_enrollment"], t1["label_safety"]) == (1, None)   # terminated for enrollment
    assert (t2["label_enrollment"], t2["label_safety"]) == (0, 0)      # completed: negative for both
    assert (t5["label_enrollment"], t5["label_safety"]) == (None, None)  # business: out of both
    assert (t3["label_enrollment"], t3["label_safety"]) == (None, None)  # active


def test_titles_are_for_display_only(db):
    cols = {c[0] for c in db.execute("SELECT * FROM TRIAL_FEATURES LIMIT 0").description}
    assert "brief_title" not in cols


def test_trials_started_after_2020_are_labeled_but_kept_out_of_training_splits(db):
    assert features(db, "T9")["split"] == "later"
    assert features(db, "T6")["split"] == "recent"
