"""The public lookup app, on a two-trial fixture (no GCS)."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


def _load(name: str, path: Path):
    """Import a folder's main.py under its own name: lookup_app and ingest_job both have a main.py."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


main = _load("lookup_app_main", Path(__file__).resolve().parents[1] / "lookup_app" / "main.py")

REASONS = [{"feature": "healthy_volunteers", "text": "Accepts healthy volunteers: No (raises risk)", "contribution": 0.19},
           {"feature": "phase", "text": "Phase: 1 (lowers risk)", "contribution": -0.12}]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    folder = tmp_path_factory.mktemp("serving")
    pd.DataFrame({
        "NCT_ID": ["NCT03801083", "NCT00000001"], "BRIEF_TITLE": ["Drug A in PAH", "Old trial"],
        "PHASE": ["PHASE2", "PHASE1"], "SPONSOR_NAME": ["Acme", "State U"], "SPONSOR_CLASS": ["INDUSTRY", "OTHER"],
        "STATUS": ["RECRUITING", "COMPLETED"], "START_DATE": pd.to_datetime(["2019-01-01", "2010-03-01"]).date,
        "MODEL_VERSION": ["v4", "v4"], "SCORE_TYPE": ["forward", "out_of_fold"], "RISK_SCORE": [0.61, 0.08],
        "RISK_PERCENTILE": [97.5, 20.0], "ENROLLMENT_RISK_SCORE": [0.2, 0.01],
        "REASONS": [json.dumps(REASONS), "[]"],
        "YEARS_RUNNING": [3.2, None], "NEXT_2Y_RISK": [0.084, None], "NEXT_2Y_PERCENTILE": [88.0, None],
    }).to_parquet(folder / "lookup_v4.parquet")
    (folder / "current.json").write_text(json.dumps({
        "model_version": "v4", "lookup_file": "lookup_v4.parquet", "trained_at": "2026-10-02", "git": "abc",
        "repo_url": "https://github.com/x/y", "published_at": "2026-10-02 10:00 UTC",
        "test": {"roc_auc": 0.7033, "roc_auc_ci95": [0.6901, 0.7156], "n": 10242, "base_rate": 0.1477,
                 "precision_top_10pct": 0.3047, "lift_top_10pct": 2.1},
        "enrollment_auc": 0.783, "trials": {"forward": 1, "out_of_fold": 1}, "left_out": ["us_only"],
        "audit": {"auc_change": 0.0017, "ci95": [-0.0009, 0.0044], "n": 18568, "archives": 25},
        "survival": {"version": "survival v1", "time_auc_2y": 0.6459, "time_auc_2y_yes_no": 0.6138,
                     "time_auc_2y_ci95": [0.6254, 0.6657], "gain_2y_ci95": [0.0093, 0.0538],
                     "terminated_2y_by_start": {"2013-2014": 0.0505, "2019-2020": 0.0743}}}))
    main.STORE.load(str(folder))
    return TestClient(main.app)


def test_trial_page_shows_score_rank_and_plain_reasons(client):
    r = client.get("/trial/NCT03801083")
    assert r.status_code == 200
    assert "Drug A in PAH" in r.text and "Riskier than 98% of active trials" in r.text
    assert "Accepts healthy volunteers: No" in r.text and "raises risk" in r.text
    assert "not causes" in r.text


def test_lookup_accepts_loose_input_and_redirects(client):
    r = client.get("/trial", params={"nct": " nct03801083 "}, follow_redirects=False)
    assert r.status_code in (302, 303, 307) and r.headers["location"] == "/trial/NCT03801083"
    assert client.get("/trial", params={"nct": "03801083"}, follow_redirects=False).headers["location"] == "/trial/NCT03801083"


def test_finished_trials_say_how_they_were_scored(client):
    assert "trained without" in client.get("/trial/NCT00000001").text


def test_the_top_percentile_never_claims_100_percent(client):
    main.STORE.trials.loc["NCT03801083", "risk_percentile"] = 100.0
    text = client.get("/trial/NCT03801083").text
    main.STORE.trials.loc["NCT03801083", "risk_percentile"] = 97.5
    assert "riskiest 1%" in text and "Riskier than 100%" not in text


def test_unknown_trial_is_a_helpful_404(client):
    r = client.get("/trial/NCT99999999")
    assert r.status_code == 404 and "Phase 1" in r.text


def test_json_api_and_health(client):
    body = client.get("/api/trials/NCT03801083").json()
    assert body["risk_percentile"] == 97.5 and body["reasons"][0]["contribution"] == 0.19
    assert client.get("/api/trials/NCT99999999").status_code == 404
    assert client.get("/health").json() == {"status": "ok", "trials": 2, "model_version": "v4"}


def test_home_page_states_the_validated_numbers(client):
    text = client.get("/").text
    assert "0.703" in text and "18,568" in text


def test_running_trials_show_the_next_two_years_and_finished_ones_do_not(client):
    running = client.get("/trial/NCT03801083").text
    assert "next 2 years" in running and "8%" in running and "3.2 years" in running and "88%" in running
    assert "recalibrated to termination rates" in running
    assert "next 2 years" not in client.get("/trial/NCT00000001").text


def test_home_page_reports_the_survival_model(client):
    text = client.get("/").text
    assert "0.646" in text and "5.1%" in text and "7.4%" in text


def test_json_for_a_finished_trial_has_nulls_not_nans(client):
    body = client.get("/api/trials/NCT00000001").json()
    assert body["next_2y_risk"] is None and body["years_running"] is None


def test_a_trial_not_yet_recruiting_is_not_described_as_running(client):
    main.STORE.trials.loc["NCT03801083", "years_running"] = 0.0
    text = client.get("/trial/NCT03801083").text
    main.STORE.trials.loc["NCT03801083", "years_running"] = 3.2
    assert "once it starts" in text and "has been running" not in text


def test_home_page_lists_example_trials_that_are_in_the_published_file(client, monkeypatch):
    monkeypatch.setattr(main, "EXAMPLES", [("Running now", ["NCT03801083", "NCT99999999"]), ("Gone", ["NCT88888888"])])
    text = client.get("/").text
    assert 'href="/trial/NCT03801083"' in text and "Drug A in PAH" in text and "riskier than 98% of active trials" in text
    assert "NCT99999999" not in text and "Gone" not in text            # not published: left out, empty group too
