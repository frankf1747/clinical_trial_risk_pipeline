import datetime as dt

import pytest

from ctrisk.spark.point_in_time import archive_date, nearest_archive, registry_features


@pytest.mark.parametrize("name, expected", [
    ("20170101_pipe-delimited-export.zip", "2017-01-01"),
    ("2019-07-01_export.zip", "2019-07-01"),
    ("https://example.org/static/exported_files/monthly/20200301_pipe-delimited-export.zip", "2020-03-01"),
])
def test_archive_date_comes_from_the_file_name(name, expected):
    assert archive_date(name) == expected


def test_archive_date_refuses_a_name_without_a_date():
    with pytest.raises(ValueError, match="no date"):
        archive_date("aact.zip")


def test_registry_features_cover_only_cohort_trials_present_in_the_archive(aact, spark):
    cohort = spark.createDataFrame([("NCT001",), ("NCT002",), ("NCT404",)], "nct_id string")
    rows = {r.nct_id: r for r in registry_features(aact, cohort, "2013-01-01").collect()}
    assert set(rows) == {"NCT001", "NCT002"}                 # NCT404 is not in this archive
    r = rows["NCT001"]
    assert (r.archive_date, r.phase, r.masking, r.has_dmc) == (dt.date(2013, 1, 1), "PHASE2", "DOUBLE", True)
    assert (r.healthy_volunteers, r.n_countries, r.sponsor_class, r.responsible_party) == (False, 2, "INDUSTRY", "SPONSOR")
    assert "Pembrolizumab" in r.text and r.archive_status == "COMPLETED"


def test_nearest_archive_prefers_the_last_record_on_or_before_start(spark):
    seen = spark.createDataFrame([
        ("A", dt.date(2017, 1, 1), 1), ("A", dt.date(2017, 4, 1), 2), ("A", dt.date(2017, 7, 1), 3),
        ("B", dt.date(2017, 4, 1), 4), ("B", dt.date(2017, 7, 1), 5),      # registered after it started
        ("C", dt.date(2017, 1, 1), 6),                                      # start date not known
    ], "nct_id string, archive_date date, x int")
    starts = spark.createDataFrame([("A", dt.date(2017, 5, 10)), ("B", dt.date(2017, 1, 1)), ("C", None)],
                                   "nct_id string, start_date date")
    got = {r.nct_id: (r.x, r.lag_days) for r in nearest_archive(seen, starts).collect()}
    assert got == {"A": (2, -39), "B": (4, 90)}                 # C has no start date to anchor on
