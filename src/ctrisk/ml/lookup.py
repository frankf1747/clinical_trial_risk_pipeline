"""A score for every trial in the modelled population, none from a model that saw that trial's outcome.

- active trials (split 'score'): the version's final model, forward-looking;
- finished trials that started 2015 or later ('test', 'recent', 'later'): the same final model, which
  was trained on 2008-2014 starts only, so these are held out;
- finished trials that started 2008-2014 ('train'): leave-one-start-year-out cross-fitting with the
  version's parameters, text refit in each fold.

Each row also carries its percentile among active trials scored by the same version and the
contributions that moved it most, in plain English. Rows go to TRIAL_LOOKUP_SCORES, which feeds the
public lookup. TRIAL_RISK_SCORES (active trials, append-only, the backtest's basis) is left alone.
"""
import json

import numpy as np
import pandas as pd

from ctrisk.ml.model import RiskModel, collapse_text
from ctrisk.ml.text import TextFeatures
from ctrisk.serving.labels import reason_text

SCORE_TYPES = {"score": "forward", "test": "held_out", "recent": "held_out", "later": "held_out",
               "train": "out_of_fold"}


def _like(model: RiskModel) -> RiskModel:
    """An unfitted model with the same columns, parameters and text settings."""
    text = (TextFeatures(n_components=model.text.n_components, min_df=model.text.min_df,
                         max_features=model.text.max_features) if model.text else None)
    return RiskModel(model.columns, model.params, text=text)


def cross_fit(model: RiskModel, frame: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, np.ndarray, list[str]]:
    """Out-of-fold scores and contributions for the training rows: each start year scored by a model
    fitted on the other training years. Contributions have registration text collapsed to one column."""
    train = frame[frame["split"] == "train"]
    years = pd.to_datetime(train["start_date"]).dt.year
    p = pd.Series(np.nan, index=train.index)
    contrib, names = None, None
    for year in sorted(years.unique()):
        fit = train[(years != year) & y[train.index].notna()]
        held = train[years == year]
        m = _like(model).fit(fit, y[fit.index])
        p[held.index] = m.predict_proba(held)
        c, names = collapse_text(m.contributions(held), m.feature_names)
        if contrib is None:
            contrib = np.zeros((len(train), c.shape[1]))
        contrib[train.index.get_indexer(held.index)] = c
    return p, contrib, names


def score_all(model: RiskModel, frame: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """One row per frame row (same index): score_type, score, and contributions as (names, values)."""
    out = pd.DataFrame(index=frame.index, columns=["score_type", "score", "contrib"], dtype=object)
    out["score_type"] = frame["split"].map(SCORE_TYPES)
    rest = frame[frame["split"] != "train"]
    if len(rest):
        c, names = collapse_text(model.contributions(rest), model.feature_names)
        out.loc[rest.index, "score"] = model.predict_proba(rest)
        out.loc[rest.index, "contrib"] = pd.Series([dict(zip(names, row)) for row in c], index=rest.index)
    if (frame["split"] == "train").any():
        p, c, names = cross_fit(model, frame, y)
        out.loc[p.index, "score"] = p
        out.loc[p.index, "contrib"] = pd.Series([dict(zip(names, row)) for row in c], index=p.index)
    out["score"] = out["score"].astype(float)
    return out


def percentile(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Share of reference scores strictly below each score, 0-100."""
    ref = np.sort(np.asarray(reference, dtype=float))
    return np.round(100 * np.searchsorted(ref, np.asarray(scores, dtype=float), side="left") / len(ref), 1)


def reasons(contrib: dict, trial: dict, up: int = 3, down: int = 2) -> list[dict]:
    """The contributions pushing risk up most and down most, strongest first, in plain English."""
    ranked = sorted(contrib.items(), key=lambda kv: -kv[1])
    picks = [kv for kv in ranked[:up] if kv[1] > 0] + [kv for kv in ranked[::-1][:down] if kv[1] < 0]
    return [{"feature": f, "text": reason_text(f, trial.get(f), c), "contribution": round(float(c), 4)}
            for f, c in picks]


def lookup_rows(frame: pd.DataFrame, overall: pd.DataFrame, enrollment: pd.DataFrame, version: int) -> pd.DataFrame:
    active = overall.loc[frame["split"] == "score", "score"].to_numpy()
    records = frame.to_dict("records")
    return pd.DataFrame({
        "NCT_ID": frame["nct_id"].to_numpy(),
        "MODEL_VERSION": f"v{version}",
        "SCORE_TYPE": overall["score_type"].to_numpy(),
        "RISK_SCORE": overall["score"].round(4).to_numpy(),
        "RISK_PERCENTILE": percentile(overall["score"].to_numpy(), active if len(active) else overall["score"]),
        "ENROLLMENT_RISK_SCORE": enrollment["score"].round(4).to_numpy(),
        "REASONS": [json.dumps(reasons(c, t)) for c, t in zip(overall["contrib"], records)],
    })


if __name__ == "__main__":
    from snowflake.connector.pandas_tools import write_pandas

    from ctrisk.config import load_config
    from ctrisk.ml.registry import MODELS_DIR, latest, load
    from ctrisk.warehouse.snowflake import connect

    load_config()
    version = latest(MODELS_DIR)
    models, docs = load(MODELS_DIR, version)
    clone = docs["manifest"]["snowflake_clone"]          # the rows this version was trained and tested on
    with connect() as conn:
        frame = conn.cursor().execute(f"SELECT * FROM {clone}").fetch_pandas_all()
        frame.columns = frame.columns.str.lower()
        frame = frame.reset_index(drop=True)
        overall = score_all(models["label"], frame, frame["label"])
        enrollment = score_all(models["label_enrollment"], frame, frame["label_enrollment"])
        rows = lookup_rows(frame, overall, enrollment, version)
        conn.cursor().execute("CREATE TABLE IF NOT EXISTS TRIAL_LOOKUP_SCORES (NCT_ID STRING, MODEL_VERSION STRING, "
                              "SCORE_TYPE STRING, RISK_SCORE FLOAT, RISK_PERCENTILE FLOAT, "
                              "ENROLLMENT_RISK_SCORE FLOAT, REASONS STRING)")
        conn.cursor().execute("DELETE FROM TRIAL_LOOKUP_SCORES WHERE MODEL_VERSION = %s", (f"v{version}",))
        ok, _, n, _ = write_pandas(conn, rows, "TRIAL_LOOKUP_SCORES")
    counts = rows["SCORE_TYPE"].value_counts().to_dict()
    print(f"v{version}: wrote {n} lookup scores {counts}" if ok else "write failed")
