import json
from pathlib import Path

from ctrisk.ml.report import PENDING, card

V2 = Path(__file__).resolve().parents[1] / "models" / "v2"


def load():
    return [json.loads((V2 / f"{n}.json").read_text()) for n in ("metrics", "manifest", "features")]


def test_card_for_a_saved_version_marks_what_it_did_not_compute():
    text = card(*load())
    assert "0.716 (0.703–0.729)" in text and "36,286" in text and "5,055 (13.9%)" in text
    assert "`n_countries` | numeric / flag | latest record; often edited after start" in text
    assert "`faers_reports` | numeric / flag | as of the start date" in text
    assert text.count(PENDING) >= 4                     # calibration fit, stable-only, rolling, phase, year
    assert "Not run for this version" in text and "Preliminary" in text


def test_card_reports_new_analyses_audit_and_backtest_when_present():
    metrics, manifest, features = load()
    label = metrics["targets"]["label"]
    label["models"]["lightgbm"]["test"]["calibration_fit"] = {"intercept": 0.05, "slope": 0.92}
    label["models"]["lightgbm_stable_only"] = {"test": {"roc_auc": 0.681}}
    label["rolling_origin"] = [{"test_years": "2013-2014", "train_n": 20000, "n": 9000, "positives": 1300,
                                "roc_auc": 0.702, "calibration_fit": {"intercept": 0.1, "slope": 0.9}}]
    audit = {"n_cohort": 19596, "n_matched": 19000,
             "matched": {"n": 19000, "latest": {"roc_auc": 0.711}, "point_in_time": {"roc_auc": 0.69},
                         "auc_drop": 0.021, "auc_drop_ci95": [0.012, 0.03]},
             "strict": {"n": 15000, "latest": {"roc_auc": 0.71}, "point_in_time": {"roc_auc": 0.688},
                        "auc_drop": 0.022, "auc_drop_ci95": [0.01, 0.033]},
             "change_rates": {"criteria_chars": {"terminated": 0.4, "completed": 0.2, "gap": 0.2}},
             "swap_one_column": {"criteria_chars": 0.009}, "unseen_levels": {}}
    backtest = {"run_at": "2027-10-01T00:00:00", "by_version": {"v3": {
        "n": 900, "positives": 200, "roc_auc": 0.7, "roc_auc_ci95": [0.66, 0.74], "first_scored": "2026-10-01",
        "calibration_fit": {"intercept": 0.3, "slope": 0.95}}}}
    text = card(metrics, manifest, features, audit, backtest)
    assert "intercept 0.05 (0 is ideal), slope 0.92 (1 is ideal)" in text
    assert "floor under post-start edits) | 0.681 | -0.035" in text
    assert "| 2013-2014 | 20,000 | 9,000 | 1,300 | 0.702 |" in text
    assert "+0.021 (+0.012 to +0.030)" in text and "`criteria_chars` | 40.0% | 20.0% | +0.009" in text
    assert "| v3 | 2026-10-01 | 900 | 200 | 0.700 (0.660–0.740) |" in text


def test_card_says_which_archives_a_partial_audit_used():
    metrics, manifest, features = load()
    r = {"n": 5000, "latest": {"roc_auc": 0.71}, "point_in_time": {"roc_auc": 0.7}, "auc_drop": 0.01,
         "auc_drop_ci95": [0.0, 0.02]}
    audit = {"n_cohort": 19596, "n_matched": 5000, "matched": r, "strict": r, "change_rates": {},
             "swap_one_column": {}, "unseen_levels": {}, "archives": ["2017-01-03", "2017-06-08", "2017-12-14"],
             "areas_from_latest": {"area_neoplasms": 1.0}}
    text = card(metrics, manifest, features, audit)
    assert "3 AACT archives dated 2017-01-03 to 2017-12-14" in text
    assert "Partial: archives cover only part of 2017–2020" in text
    assert "disease areas kept from the latest record for 100.0% of trials" in text


def test_card_lists_predictors_left_out_after_the_audit():
    metrics, manifest, features = load()
    features = {**features, "excluded": ["n_countries", "us_only"]}
    text = card(metrics, manifest, features)
    assert "Left out because the point-in-time audit found them edited after start: `n_countries`, `us_only`" in text


def test_status_reports_a_complete_audit_instead_of_calling_the_model_preliminary():
    metrics, manifest, features = load()
    r = {"n": 18568, "latest": {"roc_auc": 0.694}, "point_in_time": {"roc_auc": 0.696}, "auc_drop": -0.0017,
         "auc_drop_ci95": [-0.0044, 0.0009]}
    audit = {"n_cohort": 19596, "n_matched": 18568, "matched": r, "strict": r, "change_rates": {},
             "swap_one_column": {}, "unseen_levels": {}, "archives": ["2017-01-03", "2020-12-30"]}
    text = card(metrics, manifest, features, audit)
    assert "**Audited on 2017–2020 starts.**" in text and "Preliminary" not in text
    assert "changes AUC by +0.002 (95% CI -0.001 to +0.004) on 18,568 trials" in text
    partial = {**audit, "archives": ["2017-01-03", "2017-12-14"]}
    assert "**Preliminary.**" in card(metrics, manifest, features, partial)


def test_survival_card_reports_time_auc_comparison_calibration_and_trend():
    from ctrisk.ml.report import survival_card
    folder = Path(__file__).resolve().parents[1] / "models" / "survival" / "v1"
    metrics, manifest = (json.loads((folder / f"{n}.json").read_text()) for n in ("metrics", "manifest"))
    text = survival_card(metrics, manifest)
    assert "# Survival model card: when and how trials end, survival v1" in text
    assert "| 2015–2016 starts | 2y | 0.646 | 0.614 |" in text
    assert "+0.009 to +0.054" in text
    assert "| 2019-2020 | 11,498 | 16.8% | 3.6% | 7.4% | 15.3% |" in text
    assert "lower bound" in text and "Recalibration to today's risk" not in text      # v1 was not recalibrated


def test_survival_card_reports_the_recalibration_and_its_landmark_backtest():
    from ctrisk.ml.report import survival_card
    folder = Path(__file__).resolve().parents[1] / "models" / "survival" / "v1"
    metrics, manifest = (json.loads((folder / f"{n}.json").read_text()) for n in ("metrics", "manifest"))
    deciles = [{"bin": b, "mean_predicted": 0.02 * (b + 1), "observed": 0.03 * (b + 1), "n": 100} for b in range(10)]
    metrics["recalibration"] = {
        "lag_years": 1.7, "window_years": 3.0, "horizon_years": 2.0,
        "by_calendar_year": [{"year": 2022, "periods_at_risk": 38893, "terminated_observed_over_expected": 1.317,
                              "completed_observed_over_expected": 0.863}],
        "backtest": [{"landmark": "2023-01-01", "running": 17016, "observed": 0.1027, "predicted_as_learned": 0.0736,
                      "predicted_recalibrated": 0.0841, "time_auc": 0.5922, "time_auc_yes_no_model": 0.556,
                      "calibration": deciles, "calibration_as_learned": deciles}],
        "serving": {"fit_window": ["2022-01-13", "2025-01-13"], "fit_periods": 118827, "params": [-1.1, 0.65, -0.15, 0.95]}}
    text = survival_card(metrics, manifest)
    assert "the served model is recalibrated for that" in text and "lower bound" not in text
    assert "| 2022 | 38,893 | 1.32 | 0.86 |" in text
    assert "terminating -1.10 + 0.65 × learned log-odds" in text
    assert "| 2023-01-01 | 17,016 | 10.3% | 7.4% | 8.4% | 0.592 | 0.556 |" in text


def test_card_reports_the_ensemble_and_its_refit_stability():
    metrics, manifest, features = load()
    agree = {"spearman": 0.931, "median_abs_change": 0.0168, "p95_abs_change": 0.0673, "median_percentile_change": 5.7,
             "share_moved_over_5_points": 0.535, "share_moved_over_10_points": 0.322}
    tight = {**agree, "spearman": 0.992, "share_moved_over_10_points": 0.024}
    metrics["targets"]["label"]["refit_stability"] = {"seeds": 10, "trials": 40136, "single_fit": agree, "ensemble": tight}
    text = card(metrics, {**manifest, "seeds": 10}, features)
    assert "an ensemble of 10 such fits" in text and "## Stability across refits" in text
    assert "| Trials moving more than 10 percentile points | 32.2% | 2.4% |" in text
    assert "| Rank correlation between the two fits | 0.931 | 0.992 |" in text
    assert "## Stability across refits" not in card(*load())


def test_survival_card_reports_refit_stability_of_the_next_two_year_risk():
    from ctrisk.ml.report import survival_card
    folder = Path(__file__).resolve().parents[1] / "models" / "survival" / "v1"
    metrics, manifest = (json.loads((folder / f"{n}.json").read_text()) for n in ("metrics", "manifest"))
    agree = {"spearman": 0.867, "median_abs_change": 0.0098, "p95_abs_change": 0.0435, "median_percentile_change": 8.24,
             "share_moved_over_5_points": 0.651, "share_moved_over_10_points": 0.43}
    metrics["refit_stability"] = {"seeds": 10, "running_trials": 26251, "single_fit": agree,
                                  "ensemble": {**agree, "spearman": 0.99, "share_moved_over_10_points": 0.04}}
    text = survival_card(metrics, manifest)
    assert "the 26,251 trials running at the snapshot" in text
    assert "| Rank correlation between the two fits | 0.867 | 0.990 |" in text
    assert "| Trials moving more than 10 percentile points | 43.0% | 4.0% |" in text
