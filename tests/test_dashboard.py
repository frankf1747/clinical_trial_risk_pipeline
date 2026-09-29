import json

import pandas as pd

from ctrisk.serving.dashboard import payload, render


def target(auc, base=0.7, ci=(0.6, 0.8)):
    lg = {"roc_auc": auc, "roc_auc_ci95": list(ci), "pr_auc": 0.3, "precision_top_10pct": 0.34,
          "base_rate": 0.15, "n": 1000, "calibration": [{"bin": 0, "mean_predicted": 0.1, "observed": 0.12, "n": 500}]}
    return {"models": {"lightgbm": {"test": lg}, "logistic_regression": {"test": {"roc_auc": base}},
                       "lightgbm_no_text": {"test": {"roc_auc": auc - 0.02}},
                       "lightgbm_no_burden": {"test": {"roc_auc": auc - 0.01}},
                       "lightgbm_no_faers": {"test": {"roc_auc": auc - 0.001}}},
            "by_sponsor_class": {"INDUSTRY": {"n": 600, "roc_auc": 0.74}, "OTHER": {"n": 400, "roc_auc": 0.67}}}


METRICS = {"targets": {"label": target(0.716), "label_enrollment": target(0.791), "label_safety": target(0.666)},
           "top_drivers": [{"feature": "registration_text", "mean_abs_contribution": 0.42}]}
MANIFEST = {"version": 2, "git": "f00e8ad", "trained_at": "2026-09-28T17:53:00+00:00", "snowflake_clone": "TRIAL_FEATURES_V2"}
ACTIVE = pd.DataFrame({
    "nct_id": ["NCT1", "NCT2", "NCT3"], "brief_title": ["Low risk", "High risk </script>", None],
    "phase": ["PHASE1", "PHASE2", "PHASE3"], "sponsor_class": ["OTHER", "INDUSTRY", "OTHER"],
    "disease_area": ["Oncology", "Oncology", "Other"], "risk_score": [0.05, 0.40, 0.20], "risk_decile": [1, 10, 6],
    "enrollment_risk_score": [0.01, 0.15, 0.05], "top_driver_1": ["phase", "registration_text", "us_only"],
    "top_driver_2": [None, "us_only", None], "top_driver_3": [None, None, None],
    "model_version": ["v2"] * 3, "scored_at": ["2026-09-28 17:55:00"] * 3,
})
AREAS = pd.DataFrame({"disease_area": ["Oncology", "Other"], "trials": [2, 1], "avg_risk": [0.225, 0.2],
                      "avg_enrollment_risk": [0.08, 0.05], "top_decile_trials": [1, 0]})


def test_payload_lists_riskiest_first_and_keeps_only_real_drivers():
    data = payload(ACTIVE, AREAS, METRICS, MANIFEST, top_n=2)
    assert [t["nct"] for t in data["trials"]] == ["NCT2", "NCT3"]
    assert data["trials"][0]["drivers"] == ["registration_text", "us_only"]
    s = data["summary"]
    assert (s["n_active"], s["n_listed"], s["model_version"]) == (3, 2, "v2")
    assert s["targets"]["label"]["auc"] == 0.716 and s["targets"]["label"]["positives"] == 150
    assert s["ablations"] == {"text": 0.02, "burden": 0.01, "faers": 0.001}
    assert [b["key"] for b in s["by_sponsor"]] == ["INDUSTRY", "OTHER"]


def test_render_embeds_data_safely():
    html = render(payload(ACTIVE, AREAS, METRICS, MANIFEST))
    assert html.startswith("<!doctype html>") and "<title>Clinical Trial Risk</title>" in html
    assert "__DATA__" not in html and "High risk <\\/script>" in html   # a title cannot close the script
    blob = html.split("const DATA = ", 1)[1].split(";\n", 1)[0].replace("<\\/", "</")
    assert json.loads(blob)["summary"]["targets"]["label_enrollment"]["auc"] == 0.791


def test_fragment_for_hosted_viewers_has_no_document_shell():
    assert not render(payload(ACTIVE, AREAS, METRICS, MANIFEST), standalone=False).startswith("<!doctype")
