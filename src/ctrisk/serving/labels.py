"""Model features in plain English, for people who never saw the feature table.

value_text('healthy_volunteers', False) -> 'Accepts healthy volunteers: No'. A reader must never have to
guess what a value means; contributions describe the model, so reason_text says only which way it moved
the score, not why a trial would stop.
"""
import numbers

import numpy as np
import pandas as pd

AREAS = {"neoplasms": "cancer", "cardiovascular": "cardiovascular", "nervous_system": "neurology",
         "mental": "psychiatry", "infections": "infectious disease", "respiratory": "respiratory",
         "digestive": "digestive", "metabolic": "metabolic", "immune": "immunology", "skin": "dermatology",
         "musculoskeletal": "musculoskeletal", "urogenital": "urogenital", "blood": "hematology",
         "endocrine": "endocrine"}

# feature -> (label, kind). kind decides how a value is shown: flag, share, count, years, category.
LABELS = {
    "phase": ("Phase", "phase"),
    "number_of_arms": ("Number of arms", "count"),
    "allocation": ("Allocation", "category"),
    "intervention_model": ("Design", "category"),
    "primary_purpose": ("Primary purpose", "category"),
    "masking": ("Blinding", "category"),
    "sponsor_class": ("Sponsor type", "sponsor"),
    "has_dmc": ("Has a data monitoring committee", "flag"),
    "min_age_years": ("Minimum age", "years"),
    "max_age_years": ("Maximum age", "years"),
    "healthy_volunteers": ("Accepts healthy volunteers", "flag"),
    "sex": ("Sex", "category"),
    "responsible_party": ("Responsible party", "category"),
    "n_collaborators": ("Collaborators", "count"),
    "n_keywords": ("Keywords listed", "count"),
    "sponsor_prior_trials": ("Sponsor's past trials", "count"),
    "sponsor_prior_termination_rate": ("Sponsor's past termination rate", "share"),
    "sponsor_trials_started_2y": ("Trials the sponsor started in the prior 2 years", "count"),
    "n_substances": ("Drugs with FDA adverse-event history", "count"),
    "faers_reports": ("Drug's FAERS reports before start", "count"),
    "faers_reports_12m": ("Drug's FAERS reports in the prior 12 months", "count"),
    "faers_serious_share": ("Drug's FAERS reports marked serious", "share"),
    "faers_death_share": ("Drug's FAERS reports with a death", "share"),
    "has_faers_history": ("Drug has FAERS history", "flag"),
    # dropped from v4 (POST_START_LEAKS) but kept so older versions' reasons still read
    "n_countries": ("Countries", "count"),
    "us_only": ("US-only sites", "flag"),
    "criteria_count": ("Eligibility criteria", "count"),
    "criteria_chars": ("Length of eligibility criteria (characters)", "count"),
    **{f"area_{k}": (f"Disease area, {v}", "flag") for k, v in AREAS.items()},
}
TEXT = "Wording of the registration (title, summary, criteria)"
SPONSORS = {"INDUSTRY": "industry", "OTHER": "academic or other", "GOVERNMENT": "government"}


def _missing(value) -> bool:
    return value is None or value == "missing" or (not isinstance(value, str) and pd.isna(value))


def _shown(kind: str, value) -> str:
    if _missing(value):
        return "not given"
    if kind == "flag":
        return "Yes" if value in (True, np.True_, "True", "Yes", 1) else "No"
    if kind == "phase":
        return str(value).replace("PHASE", "")
    if kind == "sponsor":
        return SPONSORS.get(str(value), str(value).lower())
    if kind == "share" and isinstance(value, numbers.Number):
        return f"{float(value) * 100:.{0 if float(value) * 100 == round(float(value) * 100) else 1}f}%"
    if kind in ("count", "years") and isinstance(value, numbers.Number):
        return f"{float(value):g}"
    return str(value).replace("_", " ").lower()


def value_text(feature: str, value) -> str:
    if feature == "registration_text":
        return TEXT
    label, kind = LABELS.get(feature, (feature.replace("_", " ").capitalize(), "category"))
    return f"{label}: {_shown(kind, value)}"


def reason_text(feature: str, value, contribution: float) -> str:
    return f"{value_text(feature, value)} ({'raises' if contribution > 0 else 'lowers'} risk)"
