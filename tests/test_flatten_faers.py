import json
import zipfile

from ctrisk.spark.flatten_faers import build_drug_events, report_rows


def report(rid, version="1", serious="1", death=None, drugs=()):
    r = {"safetyreportid": rid, "safetyreportversion": version, "receivedate": "20160107",
         "receiptdate": "20160107", "serious": serious, "patient": {"drug": list(drugs)}}
    if death:
        r["seriousnessdeath"] = death
    return r


ONDANSETRON = {"drugcharacterization": "1", "medicinalproduct": "ZOFRAN",
               "openfda": {"substance_name": ["ONDANSETRON HYDROCHLORIDE", "ONDANSETRON"]}}
NO_OPENFDA = {"drugcharacterization": "2", "medicinalproduct": "Some Brand",
              "activesubstance": {"activesubstancename": "METFORMIN HYDROCHLORIDE"}}
PRODUCT_ONLY = {"drugcharacterization": "1", "medicinalproduct": "Grandma's Tonic"}


def test_salt_variants_collapse_to_one_substance():
    rows = list(report_rows(report("1", drugs=[ONDANSETRON])))
    assert rows == [("1", 1, "20160107", "20160107", True, False, True, True, False, "ondansetron")]


def test_falls_back_to_active_substance_and_flags_it():
    rows = list(report_rows(report("2", serious="2", death="1", drugs=[NO_OPENFDA])))
    assert rows == [("2", 1, "20160107", "20160107", False, True, False, False, True, "metformin")]


def test_product_name_is_kept_but_flagged_as_free_text():
    rows = list(report_rows(report("3", drugs=[PRODUCT_ONLY])))
    assert rows == [("3", 1, "20160107", "20160107", True, False, True, False, False, "grandma s tonic")]


def test_skips_reports_with_no_id_or_null_drug_list():
    assert list(report_rows({"safetyreportid": "5", "patient": {"drug": None}})) == []
    assert list(report_rows({"patient": {"drug": [ONDANSETRON]}})) == []


def write_zip(path, reports):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("drug-event.json", json.dumps({"meta": {}, "results": reports}))


def test_keeps_latest_report_version_only(spark, tmp_path):
    (tmp_path / "2016q1").mkdir()
    write_zip(tmp_path / "2016q1" / "a.zip", [report("9", version="1", serious="2", drugs=[ONDANSETRON])])
    write_zip(tmp_path / "2016q1" / "b.zip", [report("9", version="2", serious="1", drugs=[ONDANSETRON]),
                                              report("10", drugs=[NO_OPENFDA])])

    rows = {r.safetyreportid: r for r in build_drug_events(spark, str(tmp_path / "*" / "*.zip")).collect()}

    assert set(rows) == {"9", "10"}
    assert rows["9"].serious is True               # version 2 wins
    assert str(rows["9"].receivedate) == "2016-01-07"
    assert str(rows["9"].receiptdate) == "2016-01-07"
    assert rows["10"].substance == "metformin"
