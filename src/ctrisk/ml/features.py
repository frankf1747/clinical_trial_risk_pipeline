"""Which TRIAL_FEATURES columns the model sees, and how they become a matrix."""
import numpy as np
import pandas as pd

NOT_INPUTS = {"nct_id", "label", "label_enrollment", "label_safety", "split", "start_date", "text", "embedding",
              "end_date"}
CATEGORICAL = {"phase", "allocation", "intervention_model", "primary_purpose", "masking",
               "sponsor_class", "sex", "responsible_party"}
FAERS = ["n_substances", "faers_reports", "faers_reports_12m", "faers_serious_share",
         "faers_death_share", "has_faers_history"]
BURDEN = ["n_countries", "us_only", "min_age_years", "max_age_years", "healthy_volunteers",
          "criteria_count", "criteria_chars"]
# Registry fields most likely to be edited after a trial starts: eligibility amendments, sites and
# countries added or dropped, collaborators, keywords and oversight changed. The registration text is
# edit-prone too (summaries and criteria get rewritten). The model without these, and without text, puts
# a floor under how much of the AUC could come from post-start edits, before any archive is checked.
EDIT_PRONE = ["criteria_count", "criteria_chars", "n_countries", "us_only", "n_collaborators", "n_keywords",
              "responsible_party", "has_dmc"]

# The point-in-time audit (docs/audits/2026-10-01-point-in-time.md) found these four changed after start
# more often for trials that went on to terminate (gaps of 5-10 points), and together they carry the whole
# measured AUC inflation. Archives start in 2017, so they cannot be rebuilt as of start for the training
# years; from v4 the model leaves them out.
POST_START_LEAKS = ["criteria_count", "criteria_chars", "n_countries", "us_only"]


def inputs(frame: pd.DataFrame, drop=()) -> list[str]:
    return [c for c in frame.columns if c not in NOT_INPUTS and c not in set(drop)]


def to_matrix(frame: pd.DataFrame, columns: list[str],
              categories: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Categoricals get a 'missing' level and fixed levels; everything else becomes float."""
    X = pd.DataFrame(index=frame.index)
    learned = {}
    for c in columns:
        if c in CATEGORICAL:
            values = frame[c].astype(object).where(frame[c].notna(), "missing").astype(str)
            levels = categories[c] if categories else sorted(values.unique())
            X[c] = pd.Categorical(values, categories=levels)
            learned[c] = levels
        else:
            X[c] = frame[c].map(lambda v: np.nan if pd.isna(v) else float(v))  # bool/Decimal/NA -> float
    return X, learned
