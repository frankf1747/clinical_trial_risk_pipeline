import duckdb
import pytest

from ctrisk.warehouse.sql import failed_checks, statements


def test_splits_on_line_ending_semicolons_and_skips_comment_only_chunks(tmp_path):
    f = tmp_path / "a.sql"
    f.write_text("-- header\nCREATE TABLE t (x INT);\n\nINSERT INTO t VALUES (1);\n-- trailing note\n")
    assert statements(f) == ["-- header\nCREATE TABLE t (x INT)", "INSERT INTO t VALUES (1)"]


def test_named_checks_report_rows_that_break_a_rule(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1), (1), (2)) v(x)")
    f = tmp_path / "checks.sql"
    f.write_text("-- check: x is unique\nSELECT x FROM t GROUP BY x HAVING COUNT(*) > 1;\n"
                 "-- check: x is positive\nSELECT x FROM t WHERE x <= 0;\n")
    assert failed_checks(con, f) == ["x is unique: 1 rows, e.g. [(1,)]"]


def test_a_statement_without_a_check_name_raises_a_clear_error(tmp_path):
    con = duckdb.connect()
    f = tmp_path / "checks.sql"
    f.write_text("SELECT 1;\n")
    with pytest.raises(ValueError, match=r"checks\.sql: every check needs a '-- check: <name>' line"):
        failed_checks(con, f)
