import json
from pathlib import Path

from ctrisk.serving.publish import summary

V4 = Path(__file__).resolve().parents[1] / "models" / "v4"


def test_summary_carries_what_the_app_shows():
    docs = {n: json.loads((V4 / f"{n}.json").read_text()) for n in ("metrics", "manifest", "audit_point_in_time")}
    s = summary(docs["metrics"], docs["manifest"], docs["audit_point_in_time"],
                counts={"forward": 29894, "held_out": 45000, "out_of_fold": 36000}, lookup_file="lookup_v4.parquet")
    assert s["model_version"] == "v4" and s["lookup_file"] == "lookup_v4.parquet"
    assert s["test"]["roc_auc"] == 0.7033 and s["test"]["roc_auc_ci95"] == [0.6901, 0.7156]
    assert s["test"]["lift_top_10pct"] == round(0.3047 / 0.1477, 1)
    assert s["audit"]["auc_change"] == 0.0017 and s["audit"]["n"] == 18568
    assert s["trials"] == {"forward": 29894, "held_out": 45000, "out_of_fold": 36000}
    assert s["left_out"] == ["criteria_count", "criteria_chars", "n_countries", "us_only"]


def test_summary_includes_the_survival_model_when_there_is_one():
    docs = {n: json.loads((V4 / f"{n}.json").read_text()) for n in ("metrics", "manifest")}
    sv = V4.parent / "survival" / "v1"
    survival = {n: json.loads((sv / f"{n}.json").read_text()) for n in ("metrics", "manifest")}
    s = summary(docs["metrics"], docs["manifest"], None, counts={}, lookup_file="x", survival=survival)
    assert s["survival"]["time_auc_2y"] == 0.6459 and s["survival"]["gain_2y_ci95"] == [0.0093, 0.0538]
    assert s["survival"]["terminated_2y_by_start"]["2019-2020"] == 0.0743
