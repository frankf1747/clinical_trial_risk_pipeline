import numpy as np
import pandas as pd

from ctrisk.ml import train_survival
from ctrisk.ml.survival import SurvivalModel


class Constant:
    def predict_proba(self, frame):
        return frame["x"].to_numpy()


def trials(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    start = pd.to_datetime(rng.choice(pd.date_range("2008-01-01", "2020-12-31", freq="D"), n))
    years = rng.exponential(1 / (0.25 * np.exp(0.8 * x)))
    comp = rng.exponential(2.0, size=n)
    end = start + pd.to_timedelta(np.minimum(years, comp) * 365.25, unit="D")
    running = end > pd.Timestamp("2026-09-26")
    label = np.where(running, np.nan, (years < comp).astype(float))
    split = np.select([running, start.year < 2015, start.year < 2017], ["score", "train", "test"], "recent")
    return pd.DataFrame({"nct_id": [f"N{i}" for i in range(n)], "split": split, "start_date": start.date,
                         "end_date": np.where(running, None, end.date), "label": label, "x": x,
                         "phase": rng.choice(["PHASE1", "PHASE2"], n), "text": np.where(x > 1, "slow", "big")})


def test_run_fits_on_pre_2015_starts_and_reports_both_cohorts(monkeypatch):
    seen = []
    original = SurvivalModel.fit
    monkeypatch.setattr(SurvivalModel, "fit", lambda self, f, t, e, valid=None: seen.append(f["start"].max()) or original(self, f, t, e, valid))
    grid = [{"num_leaves": 7, "learning_rate": 0.2, "min_child_samples": 20}]
    _, report = train_survival.run(trials(), ["x", "phase"], Constant(), grid=grid, text_min_df=1)
    assert all(s < pd.Timestamp("2015-01-01") for s in seen)
    assert set(report["cohorts"]) == {"test", "recent"}
    two = report["cohorts"]["test"]["by_horizon"]["2y"]
    assert two["time_auc"] > 0.65 and two["brier"] < two["brier_null"]
    assert len(report["cohorts"]["recent"]["calibration_2y"]) == 10
