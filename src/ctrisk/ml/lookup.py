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

from ctrisk.ml.embed import EmbeddingFeatures, attach
from ctrisk.ml.model import RiskModel, collapse_text
from ctrisk.ml.text import TextFeatures
from ctrisk.serving.labels import reason_text

EMBEDDINGS = "data/parquet/trial_embeddings"
SCORE_TYPES = {"score": "forward", "test": "held_out", "recent": "held_out", "later": "held_out",
               "train": "out_of_fold"}


def _like(model: RiskModel) -> RiskModel:
    """An unfitted model with the same columns, parameters, text settings and ensemble size."""
    text = (TextFeatures(n_components=model.text.n_components, min_df=model.text.min_df,
                         max_features=model.text.max_features) if model.text else None)
    embed = getattr(model, "embed", None)
    return RiskModel(model.columns, model.params, text=text,
                     embed=EmbeddingFeatures(n_components=embed.n_components) if embed else None,
                     seeds=getattr(model, "seeds", 1))


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


MAX_ELAPSED = 6.0     # the survival model sees 8 years; a 2-year window must end inside that


def years_running(start: pd.Series, status: pd.Series) -> pd.Series:
    """Years from start to the snapshot; 0 for trials that have not started (future start date) or have not
    begun recruiting even though their planned start has passed."""
    from ctrisk.ml.survival import SNAPSHOT
    years = ((pd.Timestamp(SNAPSHOT) - pd.to_datetime(start)).dt.days / 365.25).clip(lower=0)
    return years.where(status.to_numpy() != "NOT_YET_RECRUITING", 0.0)


def next_two_years(years_running: pd.Series, risk: pd.Series) -> pd.DataFrame:
    """For running trials: years running, risk of termination in the next 2 years, and its percentile among
    running trials. Empty for finished trials and for trials past MAX_ELAPSED years."""
    years = years_running.round(1)                       # decide on the value that is stored and shown
    ok = years.notna() & (years <= MAX_ELAPSED) & risk.notna()
    out = pd.DataFrame({"years_running": years, "next_2y": risk.where(ok).round(4)})
    out["next_2y_percentile"] = np.nan
    if ok.any():
        out.loc[ok, "next_2y_percentile"] = percentile(risk[ok].to_numpy(), risk[ok].to_numpy())
    return out


def lookup_rows(frame: pd.DataFrame, overall: pd.DataFrame, enrollment: pd.DataFrame, version: int,
                next_2y: pd.DataFrame | None = None) -> pd.DataFrame:
    active = overall.loc[frame["split"] == "score", "score"].to_numpy()
    records = frame.to_dict("records")
    nxt = next_2y if next_2y is not None else pd.DataFrame(
        np.nan, index=frame.index, columns=["years_running", "next_2y", "next_2y_percentile"])
    return pd.DataFrame({
        "NCT_ID": frame["nct_id"].to_numpy(),
        "MODEL_VERSION": f"v{version}",
        "SCORE_TYPE": overall["score_type"].to_numpy(),
        "RISK_SCORE": overall["score"].round(4).to_numpy(),
        "RISK_PERCENTILE": percentile(overall["score"].to_numpy(), active if len(active) else overall["score"]),
        "ENROLLMENT_RISK_SCORE": enrollment["score"].round(4).to_numpy(),
        "REASONS": [json.dumps(reasons(c, t)) for c, t in zip(overall["contrib"], records)],
        "YEARS_RUNNING": nxt["years_running"].reindex(frame.index).to_numpy(),
        "NEXT_2Y_RISK": nxt["next_2y"].reindex(frame.index).to_numpy(),
        "NEXT_2Y_PERCENTILE": nxt["next_2y_percentile"].reindex(frame.index).to_numpy(),
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
        frame = conn.cursor().execute(f"SELECT * FROM {clone} ORDER BY nct_id").fetch_pandas_all()
        frame.columns = frame.columns.str.lower()
        frame = frame.reset_index(drop=True)
        if getattr(models["label"], "embed", None):                # M9 models read text embeddings too
            frame = attach(frame, EMBEDDINGS)
        overall = score_all(models["label"], frame, frame["label"])
        enrollment = score_all(models["label_enrollment"], frame, frame["label_enrollment"])
        nxt = None
        if any((MODELS_DIR / "survival").glob("v*")):              # M8: time to termination for running trials
            survival, _ = load(MODELS_DIR / "survival", latest(MODELS_DIR / "survival"))
            running = frame[frame["split"] == "score"]
            status = dict(conn.cursor().execute("SELECT nct_id, status FROM RAW_TRIALS").fetchall())
            years = years_running(running["start_date"], running["nct_id"].map(status))
            risk = pd.Series(survival.conditional_cif(running, years.to_numpy(), window=2.0), index=running.index)
            nxt = next_two_years(years.reindex(frame.index), risk.reindex(frame.index))
        rows = lookup_rows(frame, overall, enrollment, version, next_2y=nxt)
        conn.cursor().execute("CREATE TABLE IF NOT EXISTS TRIAL_LOOKUP_SCORES (NCT_ID STRING, MODEL_VERSION STRING, "
                              "SCORE_TYPE STRING, RISK_SCORE FLOAT, RISK_PERCENTILE FLOAT, "
                              "ENROLLMENT_RISK_SCORE FLOAT, REASONS STRING)")
        for col in ("YEARS_RUNNING", "NEXT_2Y_RISK", "NEXT_2Y_PERCENTILE"):
            conn.cursor().execute(f"ALTER TABLE TRIAL_LOOKUP_SCORES ADD COLUMN IF NOT EXISTS {col} FLOAT")
        conn.cursor().execute("DELETE FROM TRIAL_LOOKUP_SCORES WHERE MODEL_VERSION = %s", (f"v{version}",))
        ok, _, n, _ = write_pandas(conn, rows, "TRIAL_LOOKUP_SCORES")
    counts = rows["SCORE_TYPE"].value_counts().to_dict()
    print(f"v{version}: wrote {n} lookup scores {counts}" if ok else "write failed")
