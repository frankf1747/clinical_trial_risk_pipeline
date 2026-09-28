"""Which TRIAL_FEATURES columns the model sees, and how they become a matrix."""
import numpy as np
import pandas as pd

NOT_INPUTS = {"nct_id", "label", "label_enrollment", "label_safety", "split", "start_date", "text"}
CATEGORICAL = {"phase", "allocation", "intervention_model", "primary_purpose", "masking",
               "sponsor_class", "sex", "responsible_party"}
FAERS = ["n_substances", "faers_reports", "faers_reports_12m", "faers_serious_share",
         "faers_death_share", "has_faers_history"]
BURDEN = ["n_countries", "us_only", "min_age_years", "max_age_years", "healthy_volunteers",
          "criteria_count", "criteria_chars"]


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
