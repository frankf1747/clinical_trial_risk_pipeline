# M5: A Stronger, Reason-Aware Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do not commit** — Frank commits this repo himself. Only Task 8 touches Snowflake/GCS.

**Goal:** Raise the held-out AUC honestly and make the model answer the question reviewers actually ask — *why* a trial is at risk — by adding registration-time features, text, tuned LightGBM, and separate enrollment and safety targets.

**Architecture:** Spark adds a stop-reason label, six registration-time attributes, and one text blob per trial; Snowflake carries them into `TRIAL_FEATURES` with two extra targets. In Python, a `RiskModel` bundles categorical handling, TF-IDF→SVD text features (fit on training rows only), and LightGBM into one object that is tuned on an inner time split, evaluated per target with ablations, saved as a version, and used for scoring.

**Tech Stack:** PySpark, Snowflake SQL (DuckDB in tests), scikit-learn (TF-IDF, TruncatedSVD), LightGBM, pandas, pytest.

## Diagnosis this plan acts on

| Finding (from real data, 2026-09-27) | Response |
|---|---|
| 33% of terminations are enrollment, 23% business, 11% safety, 11% efficacy | Separate `label_enrollment` and `label_safety` targets; FAERS gets a fair test on safety |
| Academic trials: 18% termination but only 0.65 AUC (industry 0.72) | Text, DMC, responsible party, collaborators, concurrent sponsor load — the features that describe academic trials |
| DMC 19.5% vs 13.2%; investigator-sponsored 22% vs 15% | `has_dmc`, `responsible_party` |
| No text used at all; titles/summaries/criteria are written at registration | TF-IDF + SVD text features, fit on train only |
| Fixed LightGBM params, no validation | Small grid + early stopping on an inner time split (fit <2013, validate 2013–14) |
| v1: 0.693 AUC (CI 0.681–0.706) | Realistic target 0.72–0.75; **stop rule unchanged: any test AUC > 0.85 is investigated before it is believed** |

Not in this plan: planned enrollment and site counts (the record leaks the outcome; the point-in-time route via archived AACT snapshots is M6), full FAERS (S1).

---

## File map

| File | Responsibility |
|---|---|
| `src/ctrisk/spark/clean_trials.py` | + `stop_reason`, `has_dmc`; `sponsor_outcomes` gains `start_date` |
| `src/ctrisk/spark/trial_attributes.py` | + responsible party, collaborators, outcome counts, keyword count |
| `src/ctrisk/spark/trial_text.py` | One text blob per trial |
| `src/ctrisk/ingest/aact.py`, `tests/conftest.py` | + 4 AACT tables |
| `sql/10_raw_tables.sql`, `sql/11_copy.sql`, `sql/20_features/30_int_sponsor_history.sql`, `sql/20_features/40_trial_features.sql`, `sql/90_checks.sql` | New columns, concurrent-trials table, two targets, text join |
| `src/ctrisk/ml/text.py` | `TextFeatures`: TF-IDF → SVD |
| `src/ctrisk/ml/model.py` | `RiskModel`: tabular + text + LightGBM in one object |
| `src/ctrisk/ml/features.py` | New non-inputs and categoricals |
| `src/ctrisk/ml/train.py` | Tuning, targets, ablations, subgroup AUC |
| `src/ctrisk/ml/score.py` | Overall + enrollment scores |
| `tests/…` | One test file per unit; fixtures extended |
| `Makefile`, `README.md` | `make text`; run steps |

---

### Task 1: Stop reasons and two more targets

**Files:** Modify `tests/fixtures/aact/studies.txt`, `src/ctrisk/spark/clean_trials.py`, `tests/test_clean_trials.py`, `sql/10_raw_tables.sql`, `sql/20_features/40_trial_features.sql`, `sql/90_checks.sql`, `tests/test_features_sql.py`

- [ ] **Step 1: Extend the studies fixture** — replace `tests/fixtures/aact/studies.txt` with (three new columns: `why_stopped`, `has_dmc`, `brief_title`, `official_title`):

```
nct_id|study_type|overall_status|phase|start_date|number_of_arms|completion_date|why_stopped|has_dmc|brief_title|official_title
NCT001|INTERVENTIONAL|COMPLETED|PHASE2|2012-03-01|2|2014-01-01||t|Pembrolizumab in Lung Cancer|A Phase 2 Study of Pembrolizumab in Advanced Lung Cancer
NCT002|INTERVENTIONAL|TERMINATED|PHASE3|2015-06-15|2|2016-01-01|Sponsor decision due to slow enrollment|f|Adalimumab in RA|
NCT003|INTERVENTIONAL|Active, not recruiting|PHASE1|2024-01-10|1|||t|Metformin Pilot|Metformin Pilot Study
NCT004|OBSERVATIONAL|COMPLETED|NA|2012-01-01||2013-01-01|||Aspirin Registry|
NCT005|INTERVENTIONAL|COMPLETED|PHASE4|2012-01-01|2|2013-01-01||f|Aspirin Phase 4|
NCT006|INTERVENTIONAL|TERMINATED|PHASE2|2005-01-01|2|2006-01-01|Safety concerns|t|Old Trial|
NCT007|INTERVENTIONAL|WITHDRAWN|PHASE2|2012-01-01|2||Study withdrawn|f|Withdrawn Trial|
NCT008|INTERVENTIONAL|COMPLETED|PHASE1/PHASE2|2019-11-30|1|2020-06-01||f|Stent Study|
NCT009|Interventional|Terminated|Phase 1/Phase 2|2018-05-01|3|2019-01-01|Terminated by sponsor for business reasons|t|Ibuprofen Dosing|
NCT010|INTERVENTIONAL|COMPLETED|PHASE2|2021-02-01|2|2022-01-01||f|Late Trial|
```

- [ ] **Step 2: Write failing tests** — append to `tests/test_clean_trials.py`:

```python
def test_stop_reason_only_for_terminated_trials(trials):
    assert trials["NCT002"].stop_reason == "enrollment"
    assert trials["NCT009"].stop_reason == "business"
    assert trials["NCT001"].stop_reason is None


def test_has_dmc(trials):
    assert (trials["NCT001"].has_dmc, trials["NCT002"].has_dmc) == (True, False)


@pytest.mark.parametrize("text, expected", [
    ("Slow accrual", "enrollment"),
    ("Sponsor decision due to slow enrollment", "enrollment"),      # root cause wins over "sponsor"
    ("Unable to recruit eligible patients", "enrollment"),
    ("Company strategic decision; portfolio prioritization", "business"),
    ("Funding ended", "business"),
    ("Unacceptable toxicity in the first cohort", "safety"),
    ("Terminated for futility at interim analysis", "efficacy"),
    ("Lack of efficacy", "efficacy"),
    ("Safety signal and low enrollment", "safety"),                   # safety outranks enrollment
    ("PI left the institution", "other"),
    ("", None),
])
def test_stop_reason_classifier(spark, text, expected):
    from ctrisk.spark.clean_trials import stop_reason
    df = spark.createDataFrame([(text,)], "why_stopped string")
    assert df.select(stop_reason(F.col("why_stopped")).alias("r")).first().r == expected
```

Add `from pyspark.sql import functions as F` to the imports of `tests/test_clean_trials.py`.

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/test_clean_trials.py -v` → FAIL (`stop_reason` attribute / import errors).

- [ ] **Step 4: Implement** — in `src/ctrisk/spark/clean_trials.py`, add after `TRAIN_START, TRAIN_END = ...`:

```python
# Why a trial stopped, from the free-text reason. First match wins, so a specific cause
# ("safety") outranks a generic one ("sponsor decision").
STOP_REASONS = [
    ("safety", r"safety|adverse|toxic|tolerab|side effect|risk[- ]benefit"),
    ("efficacy", r"efficacy|futil|interim analys|lack of (clinical )?(benefit|effect)|no (clinical )?benefit"
                 r"|did not meet|not effective|ineffective"),
    ("enrollment", r"enrol|accru|recruit|slow|low number|insufficient (number|patients|subjects)"
                   r"|few (patients|subjects|participants)|no (patients|subjects|participants)"
                   r"|lack of (patients|subjects|participants|eligible)|unable to (recruit|identify|enrol)"
                   r"|not enough (patients|subjects)"),
    ("business", r"sponsor|business|strateg|fund|financ|budget|resource|company|portfolio|commercial"
                 r"|priorit|merger|acqui|contract"),
]


def stop_reason(col: Column) -> Column:
    text = F.lower(F.coalesce(col, F.lit("")))
    expr = F.lit("other")
    for name, pattern in reversed(STOP_REASONS):
        expr = F.when(text.rlike(pattern), name).otherwise(expr)
    return F.when(F.trim(text) == "", None).otherwise(expr)
```

In `build_trials`, add two fields to the `s = studies.select(...)` call, after `number_of_arms`:

```python
        F.col("why_stopped"),
        F.when(F.col("has_dmc").isNotNull(), F.col("has_dmc") == "t").alias("has_dmc"),
```

and change the final `return` so `stop_reason` is derived and `why_stopped` dropped:

```python
    return (s.where(eligible)
            .join(drug_trials, "nct_id")
            .withColumn("label", F.when(F.col("status") == "TERMINATED", 1)
                                  .when(F.col("status") == "COMPLETED", 0).cast("int"))
            .withColumn("stop_reason", F.when(F.col("label") == 1, stop_reason(F.col("why_stopped"))))
            .join(design, "nct_id", "left")
            .join(lead_sponsor, "nct_id", "left")
            .drop("study_type", "why_stopped"))
```

- [ ] **Step 5: Run** — `uv run pytest tests/test_clean_trials.py -v` → all pass (the existing tests still hold; the fixture only gained columns).

- [ ] **Step 6: Carry the reason into SQL as two targets**

`sql/10_raw_tables.sql` — `RAW_TRIALS` becomes:

```sql
CREATE OR REPLACE TABLE RAW_TRIALS (
    nct_id STRING, status STRING, phase STRING, start_date DATE, number_of_arms INT, label INT,
    stop_reason STRING, has_dmc BOOLEAN,
    allocation STRING, intervention_model STRING, primary_purpose STRING, masking STRING,
    sponsor_name STRING, sponsor_class STRING
);
```

`sql/20_features/40_trial_features.sql` — after the `split` expression add:

```sql
       -- Reason-specific targets: 1 = terminated for that reason, 0 = completed,
       -- NULL = terminated for another reason (left out of that model). stop_reason itself stays out.
       CASE WHEN t.label = 0 THEN 0 WHEN t.stop_reason = 'enrollment' THEN 1 END AS label_enrollment,
       CASE WHEN t.label = 0 THEN 0 WHEN t.stop_reason = 'safety'     THEN 1 END AS label_safety,
```

and add `t.has_dmc,` to the design-field line (after `t.sponsor_class,`).

`sql/90_checks.sql` — append:

```sql

-- check: reason targets are only set for completed or matching terminated trials
SELECT nct_id FROM TRIAL_FEATURES
WHERE (label_enrollment = 1 AND label <> 1) OR (label_safety = 1 AND label <> 1)
   OR ((label_enrollment = 0 OR label_safety = 0) AND label <> 0);
```

- [ ] **Step 7: Test the targets** — in `tests/test_features_sql.py`, change the `RAW_TRIALS` insert to include `stop_reason`:

```python
    con.execute("""INSERT INTO RAW_TRIALS (nct_id, start_date, label, stop_reason, sponsor_name, sponsor_class) VALUES
        ('T1', DATE '2015-06-15', 1,    'enrollment', 'Acme',  'INDUSTRY'),
        ('T2', DATE '2019-03-10', 0,    NULL,         'Acme',  'INDUSTRY'),
        ('T3', DATE '2024-01-01', NULL, NULL,         'Beta',  'OTHER'),     -- active, no matched drug
        ('T4', DATE '2010-01-15', 0,    NULL,         'Gamma', 'OTHER'),     -- its drug is only reported later
        ('T5', DATE '2020-06-01', 1,    'business',   'Acme',  'INDUSTRY'),  -- has its own outcome row
        ('T6', DATE '2017-01-01', 0,    NULL,         'Delta', 'OTHER'),     -- first day of the recent years
        ('T7', DATE '2015-01-01', 0,    NULL,         'Eps',   'OTHER'),     -- first day of the test years
        ('T8', DATE '2014-12-31', 0,    NULL,         'Eps',   'OTHER')      -- last day of the training years""")
```

and add:

```python
def test_reason_targets(db):
    t1, t2, t5, t3 = (features(db, t) for t in ("T1", "T2", "T5", "T3"))
    assert (t1["label_enrollment"], t1["label_safety"]) == (1, None)   # terminated for enrollment
    assert (t2["label_enrollment"], t2["label_safety"]) == (0, 0)      # completed: negative for both
    assert (t5["label_enrollment"], t5["label_safety"]) == (None, None)  # business: out of both
    assert (t3["label_enrollment"], t3["label_safety"]) == (None, None)  # active
```

- [ ] **Step 8: Run all** — `uv run pytest -q && uv run ruff check src tests` → all pass.

---

### Task 2: Registration-time features

**Files:** Modify `src/ctrisk/ingest/aact.py`, `tests/conftest.py`, `src/ctrisk/spark/trial_attributes.py`, `src/ctrisk/spark/clean_trials.py`, `tests/test_trial_attributes.py`, `tests/test_sponsor_outcomes.py`, `sql/10_raw_tables.sql`, `sql/20_features/30_int_sponsor_history.sql`, `sql/20_features/40_trial_features.sql`, `tests/test_features_sql.py`; create `tests/fixtures/aact/{responsible_parties,design_outcomes,keywords,brief_summaries}.txt`

- [ ] **Step 1: Add AACT tables** — in `src/ctrisk/ingest/aact.py`:

```python
TABLES = ("studies", "designs", "sponsors", "interventions",
          "intervention_other_names", "browse_interventions",
          "countries", "eligibilities", "browse_conditions",
          "responsible_parties", "design_outcomes", "keywords", "brief_summaries")
```

Use the same 13-name tuple in the `aact` fixture of `tests/conftest.py`.

- [ ] **Step 2: Fixtures**

`tests/fixtures/aact/responsible_parties.txt`
```
id|nct_id|responsible_party_type|name
1|NCT001|SPONSOR|Merck
2|NCT002|SPONSOR_INVESTIGATOR|Dr Smith
3|NCT009|PRINCIPAL_INVESTIGATOR|Dr Jones
```

`tests/fixtures/aact/design_outcomes.txt`
```
id|nct_id|outcome_type|measure|time_frame
1|NCT001|primary|Overall survival|24 months
2|NCT001|secondary|Progression-free survival|12 months
3|NCT001|secondary|Objective response rate|12 months
4|NCT002|primary|ACR20 response|24 weeks
```

`tests/fixtures/aact/keywords.txt`
```
id|nct_id|name|downcase_name
1|NCT001|lung cancer|lung cancer
2|NCT001|PD-1|pd-1
3|NCT002|rheumatoid arthritis|rheumatoid arthritis
```

`tests/fixtures/aact/brief_summaries.txt`
```
id|nct_id|description
1|NCT001|This study tests pembrolizumab in adults with advanced lung cancer.
2|NCT002|Adalimumab versus placebo in rheumatoid arthritis.
```

Also add one more collaborator row to `tests/fixtures/aact/sponsors.txt` so NCT001 has two collaborators:

```
6|NCT001|OTHER|collaborator|Some University
7|NCT001|INDUSTRY|collaborator|Another Co
```

- [ ] **Step 3: Write failing tests** — append to `tests/test_trial_attributes.py` (and extend the `attr_rows` fixture call with the four new frames: `aact["sponsors"], aact["responsible_parties"], aact["design_outcomes"], aact["keywords"]`):

```python
def test_registration_time_features(attrs):
    a = attrs["NCT001"]
    assert (a.responsible_party, a.n_collaborators, a.n_primary_outcomes, a.n_secondary_outcomes, a.n_keywords) \
        == ("SPONSOR", 2, 1, 2, 2)
    assert attrs["NCT002"].responsible_party == "SPONSOR_INVESTIGATOR"
    b = attrs["NCT003"]                                     # nothing registered beyond the study row
    assert (b.responsible_party, b.n_collaborators, b.n_primary_outcomes, b.n_keywords) == (None, 0, 0, 0)
```

Fix the existing collaborator-free assertion in `test_uses_lead_sponsor_only` if it breaks (it should not — the lead is still Merck).

- [ ] **Step 4: Run to verify failure** — `uv run pytest tests/test_trial_attributes.py -v` → FAIL (`build_trial_attributes() takes 4 positional arguments`).

- [ ] **Step 5: Implement** — in `src/ctrisk/spark/trial_attributes.py` change the signature and body:

```python
def build_trial_attributes(trials: DataFrame, countries: DataFrame, eligibilities: DataFrame,
                           browse_conditions: DataFrame, sponsors: DataFrame,
                           responsible_parties: DataFrame, design_outcomes: DataFrame,
                           keywords: DataFrame) -> DataFrame:
```

and before the final `return`, add:

```python
    party = (responsible_parties
             .select("nct_id", F.upper("responsible_party_type").alias("responsible_party"))
             .dropDuplicates(["nct_id"]))
    collab = sponsors.groupBy("nct_id").agg(
        F.sum((F.lower("lead_or_collaborator") == "collaborator").cast("int")).alias("n_collaborators"))
    outcome_type = F.lower("outcome_type")
    outcomes = design_outcomes.groupBy("nct_id").agg(
        F.sum((outcome_type == "primary").cast("int")).alias("n_primary_outcomes"),
        F.sum((outcome_type == "secondary").cast("int")).alias("n_secondary_outcomes"))
    kw = keywords.groupBy("nct_id").agg(F.count("*").alias("n_keywords"))
```

then the `return` becomes:

```python
    counts = ["n_countries", "n_collaborators", "n_primary_outcomes", "n_secondary_outcomes", "n_keywords"]
    return (trials.select("nct_id")
            .join(geo, "nct_id", "left")
            .join(elig, "nct_id", "left")
            .join(area, "nct_id", "left")
            .join(party, "nct_id", "left")
            .join(collab, "nct_id", "left")
            .join(outcomes, "nct_id", "left")
            .join(kw, "nct_id", "left")
            .fillna(0, subset=counts)
            .fillna(False, subset=["us_only", *[f"area_{k}" for k in AREAS]]))
```

Update the `__main__` block to read and pass the four extra tables:

```python
    attrs = build_trial_attributes(spark.read.parquet(cfg.path("parquet", "trials")),
                                   *[read_table(spark, aact, t) for t in
                                     ("countries", "eligibilities", "browse_conditions", "sponsors",
                                      "responsible_parties", "design_outcomes", "keywords")])
```

Update the header docstring's first line to: `"""Per-trial attributes from the registration record: geography, eligibility, disease area, who runs it, what it measures.`

- [ ] **Step 6: Sponsor start dates** — in `tests/test_sponsor_outcomes.py` extend the expected tuples with the start date:

```python
    got = {r.nct_id: (r.sponsor_name, r.terminated, str(r.start_date), str(r.completion_date)) for r in rows}
    assert got == {
        "NCT001": ("Merck", 0, "2012-03-01", "2014-01-01"),
        "NCT002": ("State University", 1, "2015-06-15", "2016-01-01"),
        "NCT009": ("Pfizer", 1, "2018-05-01", "2019-01-01"),
    }
```

Run it (fails: no `start_date`), then in `build_sponsor_outcomes` add `F.to_date("start_date").alias("start_date"),` to the `.select(...)` after `"nct_id",`. Run again → pass.

- [ ] **Step 7: SQL** — `sql/10_raw_tables.sql`:

```sql
CREATE OR REPLACE TABLE RAW_TRIAL_ATTRIBUTES (
    nct_id STRING, n_countries INT, us_only BOOLEAN, min_age_years FLOAT, max_age_years FLOAT,
    healthy_volunteers BOOLEAN, sex STRING, criteria_count INT, criteria_chars INT,
    area_neoplasms BOOLEAN, area_cardiovascular BOOLEAN, area_nervous_system BOOLEAN,
    area_mental BOOLEAN, area_infections BOOLEAN, area_respiratory BOOLEAN, area_digestive BOOLEAN,
    area_metabolic BOOLEAN, area_immune BOOLEAN, area_skin BOOLEAN, area_musculoskeletal BOOLEAN,
    area_urogenital BOOLEAN, area_blood BOOLEAN, area_endocrine BOOLEAN,
    responsible_party STRING, n_collaborators INT, n_primary_outcomes INT, n_secondary_outcomes INT,
    n_keywords INT
);

CREATE OR REPLACE TABLE RAW_SPONSOR_OUTCOMES (
    nct_id STRING, terminated INT, start_date DATE, completion_date DATE, sponsor_name STRING
);
```

`sql/20_features/30_int_sponsor_history.sql` — append:

```sql

-- The sponsor's other studies that were running on this trial's start date (started before it,
-- finished on or after it). Being under way at that date is knowable then, whatever came later.
CREATE OR REPLACE TABLE INT_SPONSOR_CONCURRENT AS
SELECT t.nct_id, COUNT(s.nct_id) AS sponsor_concurrent_trials
FROM RAW_TRIALS t
LEFT JOIN RAW_SPONSOR_OUTCOMES s
       ON s.sponsor_name = t.sponsor_name
      AND s.nct_id <> t.nct_id
      AND s.start_date < t.start_date
      AND s.completion_date >= t.start_date
GROUP BY t.nct_id;
```

`sql/20_features/40_trial_features.sql` — after `sh.sponsor_prior_termination_rate,` add `COALESCE(sc.sponsor_concurrent_trials, 0) AS sponsor_concurrent_trials,` and add the join `LEFT JOIN INT_SPONSOR_CONCURRENT sc ON sc.nct_id = t.nct_id`.

- [ ] **Step 8: Test it** — in `tests/test_features_sql.py` the sponsor insert gains a start date (positional order: nct_id, terminated, start_date, completion_date, sponsor_name):

```python
    con.execute("""INSERT INTO RAW_SPONSOR_OUTCOMES VALUES
        ('S1', 1, DATE '2013-01-01', DATE '2014-12-31', 'Acme'),   -- ended before T1
        ('S3', 1, DATE '2014-01-01', DATE '2015-06-15', 'Acme'),   -- ended ON T1's start date: concurrent, not prior
        ('S2', 0, DATE '2015-01-01', DATE '2015-06-16', 'Acme'),   -- ended the day after T1 started: concurrent
        ('T1', 1, DATE '2015-06-15', DATE '2016-01-01', 'Acme'),
        ('T5', 1, DATE '2020-06-01', DATE '2020-05-01', 'Acme')    -- T5's own row: never its own history""")
```

and add:

```python
def test_sponsor_concurrent_trials_were_running_on_the_start_date(db):
    assert features(db, "T1")["sponsor_concurrent_trials"] == 2      # S3, S2
    assert features(db, "T2")["sponsor_concurrent_trials"] == 0      # everything had finished by 2019
    assert features(db, "T3")["sponsor_concurrent_trials"] == 0
```

- [ ] **Step 9: Run all** — `uv run pytest -q && uv run ruff check src tests` → all pass. `ml/features.py` gets its update in Task 4.

---

### Task 3: Trial text

**Files:** Create `src/ctrisk/spark/trial_text.py`, `tests/test_trial_text.py`; modify `sql/10_raw_tables.sql`, `sql/11_copy.sql`, `sql/20_features/40_trial_features.sql`, `sql/90_checks.sql`, `tests/test_features_sql.py`, `Makefile`

- [ ] **Step 1: Write failing tests** — `tests/test_trial_text.py`

```python
from ctrisk.spark.clean_trials import build_trials
from ctrisk.spark.trial_text import build_trial_text


def test_one_blob_per_trial_from_registration_fields(aact):
    trials = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])
    rows = {r.nct_id: r.text for r in build_trial_text(
        trials, aact["studies"], aact["brief_summaries"], aact["eligibilities"],
        aact["design_outcomes"], aact["keywords"]).collect()}

    assert set(rows) == {"NCT001", "NCT002", "NCT003", "NCT009"}
    t = rows["NCT001"]
    assert "A Phase 2 Study of Pembrolizumab" in t          # official title preferred
    assert "advanced lung cancer" in t                       # summary
    assert "Measurable disease" in t and "~" not in t        # criteria, line breaks removed
    assert "Overall survival" in t and "Progression-free" not in t   # primary outcomes only
    assert "PD-1" in t                                       # keywords
    assert rows["NCT002"].startswith("Adalimumab in RA")     # falls back to brief title
    assert rows["NCT009"] == "Ibuprofen Dosing"              # nothing else registered
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_trial_text.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/ctrisk/spark/trial_text.py`

```python
"""One text blob per trial from fields written at registration: title, summary, eligibility
criteria, primary outcome measures, keywords. These are rarely rewritten after a trial starts."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ctrisk.config import load_config
from ctrisk.spark.clean_trials import read_table
from ctrisk.spark.session import get_spark


def build_trial_text(trials: DataFrame, studies: DataFrame, brief_summaries: DataFrame,
                     eligibilities: DataFrame, design_outcomes: DataFrame, keywords: DataFrame) -> DataFrame:
    primary = (design_outcomes.where(F.lower("outcome_type") == "primary")
               .groupBy("nct_id").agg(F.concat_ws(". ", F.collect_list("measure")).alias("outcomes")))
    kw = keywords.groupBy("nct_id").agg(F.concat_ws(", ", F.collect_list("name")).alias("keywords"))
    parts = [F.coalesce("official_title", "brief_title"), F.col("summary"),
             F.regexp_replace(F.col("criteria"), "~", " "), F.col("outcomes"), F.col("keywords")]
    return (trials.select("nct_id")
            .join(studies.select("nct_id", "official_title", "brief_title"), "nct_id", "left")
            .join(brief_summaries.select("nct_id", F.col("description").alias("summary")), "nct_id", "left")
            .join(eligibilities.select("nct_id", "criteria"), "nct_id", "left")
            .join(primary, "nct_id", "left")
            .join(kw, "nct_id", "left")
            .select("nct_id", F.concat_ws("\n", *parts).alias("text")))   # concat_ws skips nulls


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("trial_text", cfg.mode)
    aact = cfg.path("raw", "aact")
    text = build_trial_text(spark.read.parquet(cfg.path("parquet", "trials")),
                            *[read_table(spark, aact, t) for t in
                              ("studies", "brief_summaries", "eligibilities", "design_outcomes", "keywords")])
    text.write.mode("overwrite").parquet(cfg.path("parquet", "trial_text"))
    print({"trials": text.count(), "median_chars": text.select(F.expr("percentile(length(text), 0.5)")).first()[0]})
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_trial_text.py -v` → pass.

- [ ] **Step 5: SQL and wiring**

`sql/10_raw_tables.sql` — append `CREATE OR REPLACE TABLE RAW_TRIAL_TEXT (nct_id STRING, text STRING);`

`sql/11_copy.sql` — append:
```sql
COPY INTO RAW_TRIAL_TEXT FROM @PARQUET_STAGE/trial_text/
    PATTERN = '.*[.]parquet' MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
```

`sql/20_features/40_trial_features.sql` — add `x.text,` as the last selected column (after `has_faers_history`) and the join `LEFT JOIN RAW_TRIAL_TEXT x ON x.nct_id = t.nct_id`. Add to the header comment: `-- text is registration-time prose; the model turns it into features itself.`

`sql/90_checks.sql` — add `UNION ALL SELECT 'RAW_TRIAL_TEXT' FROM RAW_TRIAL_TEXT HAVING COUNT(*) = 0` to the "every RAW table loaded" check.

`tests/test_features_sql.py` — after the FAERS insert add:
```python
    con.execute("INSERT INTO RAW_TRIAL_TEXT VALUES ('T1', 'Phase 3 trial of drugx in adults'), ('T2', 'Pilot study')")
```
and a test:
```python
def test_text_is_carried_but_missing_text_is_null(db):
    assert features(db, "T1")["text"].startswith("Phase 3")
    assert features(db, "T3")["text"] is None
```

`Makefile` — add `text` to `.PHONY` and:
```make
text:
	uv run python -m ctrisk.spark.trial_text
```

- [ ] **Step 6: Run all** — `uv run pytest -q && uv run ruff check src tests` → all pass.

---

### Task 4: Text features (fit on train only)

**Files:** Create `src/ctrisk/ml/text.py`, `tests/test_ml_text.py`; modify `src/ctrisk/ml/features.py`, `tests/test_ml_features.py`

- [ ] **Step 1: Update the input rules** — `src/ctrisk/ml/features.py` constants:

```python
NOT_INPUTS = {"nct_id", "label", "label_enrollment", "label_safety", "split", "start_date", "text"}
CATEGORICAL = {"phase", "allocation", "intervention_model", "primary_purpose", "masking",
               "sponsor_class", "sex", "responsible_party"}
```

(`text` is handled by `TextFeatures`, never as a raw column; `start_date` is for splitting only.) Extend `FRAME` in `tests/test_ml_features.py` with `"label_enrollment": [1, 0, None], "text": ["a", "b", None]` and assert they are absent from `inputs(FRAME)`.

- [ ] **Step 2: Write failing tests** — `tests/test_ml_text.py`

```python
import numpy as np
import pandas as pd

from ctrisk.ml.text import TextFeatures

rng = np.random.default_rng(0)
WORDS = ["cancer", "placebo", "insulin", "pediatric", "phase", "randomized", "pain", "vaccine", "heart", "kidney"]
DOCS = pd.Series([" ".join(rng.choice(WORDS, 6)) for _ in range(300)], index=range(1000, 1300))


def test_transform_keeps_index_and_shape():
    tf = TextFeatures(n_components=4, min_df=1).fit(DOCS)
    X = tf.transform(DOCS)
    assert X.shape == (300, 4) and list(X.columns) == tf.columns == ["txt_00", "txt_01", "txt_02", "txt_03"]
    assert X.index.equals(DOCS.index)


def test_components_never_exceed_vocabulary():
    tiny = pd.Series([" ".join(rng.choice(WORDS[:3], 5)) for _ in range(100)])   # <= 12 uni/bigrams
    tf = TextFeatures(n_components=64, min_df=1).fit(tiny)
    assert tf.transform(tiny).shape[1] < 12


def test_unseen_words_and_missing_text_are_fine():
    tf = TextFeatures(n_components=4, min_df=1).fit(DOCS)
    X = tf.transform(pd.Series(["completely novel tokens", None, ""]))
    assert X.shape == (3, 4) and np.isfinite(X.to_numpy()).all()


def test_deterministic():
    a = TextFeatures(n_components=4, min_df=1).fit(DOCS).transform(DOCS)
    b = TextFeatures(n_components=4, min_df=1).fit(DOCS).transform(DOCS)
    assert np.allclose(a, b)
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/test_ml_text.py -v` → `ModuleNotFoundError`.

- [ ] **Step 4: Implement** — `src/ctrisk/ml/text.py`

```python
"""Registration text -> a few dense columns: TF-IDF over word 1-2grams, compressed with SVD.

Fit on training trials only, so nothing about test trials shapes the vocabulary or components.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatures:
    def __init__(self, n_components: int = 64, min_df: int = 20, max_features: int = 50_000):
        self.n_components, self.min_df, self.max_features = n_components, min_df, max_features
        self.tfidf = self.svd = None
        self.columns: list[str] = []

    def fit(self, texts: pd.Series) -> "TextFeatures":
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, max_features=self.max_features,
                                     sublinear_tf=True, dtype=np.float32)
        matrix = self.tfidf.fit_transform(texts.fillna(""))
        k = min(self.n_components, matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(matrix)
        self.columns = [f"txt_{i:02d}" for i in range(k)]
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        dense = self.svd.transform(self.tfidf.transform(texts.fillna("")))
        return pd.DataFrame(dense, columns=self.columns, index=texts.index)
```

- [ ] **Step 5: Run** — `uv run pytest -q && uv run ruff check src tests` → all pass.

---

### Task 5: `RiskModel`

**Files:** Create `src/ctrisk/ml/model.py`, `tests/test_ml_model.py`

- [ ] **Step 1: Write failing tests** — `tests/test_ml_model.py`

```python
import numpy as np
import pandas as pd

from ctrisk.ml.model import RiskModel
from ctrisk.ml.text import TextFeatures

PARAMS = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 10, "verbose": -1,
          "random_state": 0}


def frame(n, seed):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    return pd.DataFrame({
        "phase": rng.choice(["PHASE1", "PHASE2"], n),
        "n_countries": signal,
        "text": [("risky " if s > 0.5 else "calm ") + rng.choice(["a", "b", "c"]) for s in signal],
    }), pd.Series((signal + rng.normal(scale=0.3, size=n) > 0.8).astype(int))


def test_fit_predict_and_contributions_line_up():
    X, y = frame(600, 0)
    m = RiskModel(["phase", "n_countries"], PARAMS, text=TextFeatures(n_components=2, min_df=1)).fit(X, y)
    Xn, _ = frame(50, 1)
    p, c = m.predict_proba(Xn), m.contributions(Xn)
    assert p.shape == (50,) and (0 <= p).all() and (p <= 1).all()
    assert c.shape == (50, len(m.feature_names)) and m.feature_names == ["phase", "n_countries", "txt_00", "txt_01"]


def test_unseen_category_at_scoring_does_not_crash():
    X, y = frame(600, 0)
    m = RiskModel(["phase"], PARAMS, text=None).fit(X, y)
    assert m.categories == {"phase": ["PHASE1", "PHASE2"]}
    assert m.predict_proba(pd.DataFrame({"phase": ["PHASE9", None]})).shape == (2,)


def test_early_stopping_on_a_validation_frame():
    X, y = frame(800, 0)
    m = RiskModel(["phase", "n_countries"], {**PARAMS, "n_estimators": 500}, text=None)
    m.fit(X.iloc[:600], y.iloc[:600], valid=(X.iloc[600:], y.iloc[600:]))
    assert 1 <= m.best_iteration < 500
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_ml_model.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/ctrisk/ml/model.py`

```python
"""Tabular inputs + text features + LightGBM as one object: fit once, save with joblib, score later."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from ctrisk.ml.features import to_matrix
from ctrisk.ml.text import TextFeatures


class RiskModel:
    def __init__(self, columns: list[str], params: dict, text: TextFeatures | None, fit_text: bool = True):
        self.columns, self.params, self.text, self.fit_text = list(columns), dict(params), text, fit_text
        self.categories: dict | None = None
        self.lgbm: lgb.LGBMClassifier | None = None

    @property
    def feature_names(self) -> list[str]:
        return self.columns + (self.text.columns if self.text else [])

    @property
    def best_iteration(self) -> int:
        return self.lgbm.best_iteration_ or self.params["n_estimators"]

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        X, learned = to_matrix(frame, self.columns, self.categories)
        if self.categories is None:
            self.categories = learned
        if self.text:
            X = pd.concat([X, self.text.transform(frame["text"])], axis=1)
        return X

    def fit(self, frame: pd.DataFrame, y: pd.Series, valid: tuple | None = None) -> "RiskModel":
        if self.text and self.fit_text:
            self.text.fit(frame["text"])
        X = self._matrix(frame)                      # learns categories from the training frame only
        kwargs = {}
        if valid is not None:
            kwargs = {"eval_set": [(self._matrix(valid[0]), valid[1])],
                      "callbacks": [lgb.early_stopping(50, verbose=False)]}
        self.lgbm = lgb.LGBMClassifier(**self.params).fit(X, y, **kwargs)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.lgbm.predict_proba(self._matrix(frame))[:, 1]

    def contributions(self, frame: pd.DataFrame) -> np.ndarray:
        return self.lgbm.predict(self._matrix(frame), pred_contrib=True)[:, :-1]   # last column is bias
```

- [ ] **Step 4: Run** — `uv run pytest -q && uv run ruff check src tests` → all pass.

---

### Task 6: Training — tuning, targets, ablations, subgroups

**Files:** Rewrite `src/ctrisk/ml/train.py`, `tests/test_ml_train.py`

- [ ] **Step 1: Replace `tests/test_ml_train.py`**

```python
import numpy as np
import pandas as pd
import pytest

from ctrisk.ml.train import run

SMALL_GRID = [{"num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 20}]


def synthetic(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    frame = pd.DataFrame({
        "nct_id": [f"NCT{i:05d}" for i in range(n)],
        "split": rng.choice(["train", "test", "recent", "score"], n, p=[0.55, 0.2, 0.15, 0.1]),
        "start_date": pd.to_datetime(rng.choice(pd.date_range("2008-01-01", "2014-12-31", freq="D"), n)),
        "phase": rng.choice(["PHASE1", "PHASE2", "PHASE3"], n),
        "sponsor_class": rng.choice(["INDUSTRY", "OTHER"], n),
        "n_countries": signal,
        "faers_reports": rng.poisson(3, n),                       # noise
        "text": [("slow accrual " if s > 1.5 else "large multicenter ") + rng.choice(["a", "b"]) for s in signal],
    })
    terminated = (signal + rng.normal(scale=0.5, size=n) > 1).astype(float)
    frame["label"] = np.where(frame["split"] == "score", np.nan, terminated)
    reason = rng.choice(["enrollment", "safety", "business"], n, p=[0.6, 0.2, 0.2])
    frame["label_enrollment"] = np.where(frame["label"] == 0, 0.0,
                                         np.where((frame["label"] == 1) & (reason == "enrollment"), 1.0, np.nan))
    frame["label_safety"] = np.where(frame["label"] == 0, 0.0,
                                     np.where((frame["label"] == 1) & (reason == "safety"), 1.0, np.nan))
    return frame


@pytest.fixture(scope="module")
def result():
    return run(synthetic(), grid=SMALL_GRID, text_min_df=1)


def test_every_target_is_trained_and_evaluated(result):
    _, report = result
    assert set(report["targets"]) == {"label", "label_enrollment", "label_safety"}
    label = report["targets"]["label"]
    assert set(label["models"]) == {"logistic_regression", "lightgbm", "lightgbm_no_text",
                                    "lightgbm_no_faers", "lightgbm_no_burden"}
    assert label["models"]["lightgbm"]["test"]["roc_auc"] > 0.85
    lo, hi = label["models"]["lightgbm"]["test"]["roc_auc_ci95"]
    assert lo < label["models"]["lightgbm"]["test"]["roc_auc"] < hi
    assert label["params"]["n_estimators"] >= 50 and len(label["grid"]) == 1


def test_reason_targets_exclude_other_terminations(result):
    _, report = result
    enrol = report["targets"]["label_enrollment"]["splits"]["train"]["n"]
    all_ = report["targets"]["label"]["splits"]["train"]["n"]
    assert 0 < enrol < all_


def test_ablations_and_subgroups(result):
    _, report = result
    label = report["targets"]["label"]
    assert label["models"]["lightgbm_no_burden"]["test"]["roc_auc"] < 0.8   # n_countries is the driver
    assert set(label["by_sponsor_class"]) == {"INDUSTRY", "OTHER"}
    assert report["top_drivers"][0]["feature"] == "n_countries"


def test_returns_scoring_models_for_overall_and_enrollment(result):
    models, report = result
    assert set(models) == {"label", "label_enrollment"}
    assert report["features"]["columns"] == ["phase", "sponsor_class", "n_countries", "faers_reports"]
    assert models["label"].feature_names[:4] == report["features"]["columns"]


def test_refuses_a_train_test_or_recent_row_without_a_label():
    frame = synthetic(200)
    frame.loc[frame["split"] == "train", "label"] = np.nan
    with pytest.raises(ValueError, match="without a label"):
        run(frame, grid=SMALL_GRID, text_min_df=1)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_ml_train.py -v` → FAIL (`run()` has no `grid` argument).

- [ ] **Step 3: Rewrite `src/ctrisk/ml/train.py`**

```python
"""Train on 2008-2014 starts, evaluate on resolved 2015-2016 and censored 2017-2020, save a version.

Three targets: any termination (scored as RISK_SCORE), termination for enrollment (scored as
ENROLLMENT_RISK_SCORE), and termination for safety (reported only: it is where FAERS should matter).
For each, logistic regression is the baseline; LightGBM is tuned on an inner time split; ablations
show what text, FAERS, and burden features contribute.
"""
import subprocess
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ctrisk.ml.evaluate import bootstrap_auc_ci, calibration, metrics
from ctrisk.ml.features import BURDEN, CATEGORICAL, FAERS, inputs, to_matrix
from ctrisk.ml.model import RiskModel
from ctrisk.ml.text import TextFeatures

TARGETS = {"label": "any termination", "label_enrollment": "terminated for enrollment",
           "label_safety": "terminated for safety"}
SCORED = ("label", "label_enrollment")
BASE_PARAMS = {"n_estimators": 2000, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
               "random_state": 0, "verbose": -1}
GRID = [{"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
        for nl in (15, 31, 63) for lr in (0.03, 0.06) for mcs in (50, 200)]
TUNE_SPLIT = pd.Timestamp("2013-01-01")     # inner split: fit before, validate 2013-2014
ABLATIONS = {"lightgbm": ([], True), "lightgbm_no_text": ([], False),
             "lightgbm_no_faers": (FAERS, True), "lightgbm_no_burden": (BURDEN, True)}


def logistic_regression(columns: list[str]):
    cats = [c for c in columns if c in CATEGORICAL]
    nums = [c for c in columns if c not in CATEGORICAL]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    return make_pipeline(prep, LogisticRegression(max_iter=2000))


def tune(frame: pd.DataFrame, y: pd.Series, columns: list[str], grid: list[dict], text_min_df: int) -> tuple[dict, list]:
    """Pick LightGBM params on an inner time split of the training rows; text is fit once on the inner train."""
    inner = frame["start_date"] < TUNE_SPLIT
    text = TextFeatures(min_df=text_min_df).fit(frame.loc[inner, "text"])
    results = []
    for g in grid:
        m = RiskModel(columns, {**BASE_PARAMS, **g}, text=text, fit_text=False)
        m.fit(frame[inner], y[inner], valid=(frame[~inner], y[~inner]))
        results.append({**g, "n_estimators": int(m.best_iteration),
                        "val_auc": round(float(roc_auc_score(y[~inner], m.predict_proba(frame[~inner]))), 4)})
    best = max(results, key=lambda r: r["val_auc"])
    params = {**BASE_PARAMS, **{k: best[k] for k in ("num_leaves", "learning_rate", "min_child_samples")},
              "n_estimators": max(50, best["n_estimators"])}
    return params, results


def evaluate(p: np.ndarray, y: pd.Series, full: bool) -> dict:
    out = metrics(y, p)
    if full:
        out["roc_auc_ci95"] = bootstrap_auc_ci(y, p)
        out["calibration"] = calibration(y, p)
    return out


def train_target(frame: pd.DataFrame, target: str, columns: list[str], grid: list[dict], text_min_df: int):
    y = frame[target]
    rows = {s: (frame["split"] == s) & y.notna() for s in ("train", "test", "recent")}
    train, test, recent = (frame[rows[s]] for s in ("train", "test", "recent"))
    params, results = tune(train, y[rows["train"]], columns, grid, text_min_df)

    models, report = {}, {"description": TARGETS[target], "params": params, "grid": results, "models": {},
                          "splits": {s: {"n": int(r.sum()), "base_rate": round(float(y[r].mean()), 4)}
                                     for s, r in rows.items()}}
    X_train, cats = to_matrix(train, columns)
    lr = logistic_regression(columns).fit(X_train, y[rows["train"]])
    models["logistic_regression"] = lambda f: lr.predict_proba(to_matrix(f, columns, cats)[0])[:, 1]
    for name, (drop, use_text) in ABLATIONS.items():
        cols = [c for c in columns if c not in drop]
        text = TextFeatures(min_df=text_min_df) if use_text else None
        m = RiskModel(cols, params, text=text).fit(train, y[rows["train"]])
        models[name] = m
    for name, m in models.items():
        predict = m if callable(m) else m.predict_proba     # the LR baseline is a plain function
        report["models"][name] = {
            "test": evaluate(predict(test), y[rows["test"]], full=name == "lightgbm"),
            "recent": evaluate(predict(recent), y[rows["recent"]], full=False)}
    if "sponsor_class" in columns:
        p = pd.Series(models["lightgbm"].predict_proba(test), index=test.index)
        report["by_sponsor_class"] = {
            str(k): {"n": int(len(g)), "roc_auc": round(float(roc_auc_score(y[g.index], p[g.index])), 4)}
            for k, g in test.groupby("sponsor_class") if y[g.index].nunique() > 1}
    return models["lightgbm"], report


def run(frame: pd.DataFrame, grid: list[dict] = GRID, text_min_df: int = 20):
    columns = inputs(frame)
    labeled = frame["split"] != "score"
    if frame.loc[labeled, "label"].isna().any():
        raise ValueError("a train/test/recent row without a label; only score rows may be unlabeled")

    final, report = {}, {"targets": {}}
    for target in TARGETS:
        model, target_report = train_target(frame, target, columns, grid, text_min_df)
        report["targets"][target] = target_report
        if target in SCORED:
            final[target] = model

    main = final["label"]
    test = frame[(frame["split"] == "test") & frame["label"].notna()]
    importance = pd.Series(np.abs(main.contributions(test)).mean(axis=0), index=main.feature_names)
    report["top_drivers"] = [{"feature": f, "mean_abs_contribution": round(float(v), 4)}
                             for f, v in importance.sort_values(ascending=False).head(15).items()]
    report["features"] = {"columns": columns, "categories": main.categories,
                          "text_components": len(main.text.columns) if main.text else 0}
    return final, report


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def _git_version() -> str:
    try:
        sha, dirty = _git("rev-parse", "--short", "HEAD"), _git("status", "--porcelain")
    except OSError:  # git not installed
        return "unknown"
    return sha + ("-dirty" if dirty else "")


if __name__ == "__main__":
    from ctrisk.config import load_config
    from ctrisk.ml.registry import MODELS_DIR, next_version, save
    from ctrisk.warehouse.snowflake import connect

    load_config()  # loads .env
    version = next_version(MODELS_DIR)
    clone = f"TRIAL_FEATURES_V{version}"
    with connect() as conn, conn.cursor() as cur:
        # Frozen training data. Replace is safe: a version number is only used once its folder is saved.
        cur.execute(f"CREATE OR REPLACE TABLE {clone} CLONE TRIAL_FEATURES")
        frame = cur.execute(f"SELECT * FROM {clone}").fetch_pandas_all()
    frame.columns = frame.columns.str.lower()
    frame["start_date"] = pd.to_datetime(frame["start_date"])

    git = _git_version()  # before training, so a git problem cannot cost a finished model
    models, report = run(frame)
    manifest = {"version": version, "snowflake_clone": clone, "git": git,
                "trained_at": datetime.now(UTC).isoformat(),
                "params": {t: report["targets"][t]["params"] for t in TARGETS},
                "train": "start 2008-2014 (tuned on <2013 vs 2013-2014)", "test": "start 2015-2016",
                "recent": "start 2017-2020"}
    folder = save(MODELS_DIR, version, models, metrics={k: v for k, v in report.items() if k != "features"},
                  features=report["features"], manifest=manifest)

    print(f"saved {folder}  (training data frozen as {clone})")
    for target, t in report["targets"].items():
        print(f"\n== {target}: {t['description']}  (train n={t['splits']['train']['n']}, "
              f"base rate {t['splits']['train']['base_rate']})")
        print(f"{'model':<22}{'test AUC':>10}{'recent AUC':>12}{'test PR AUC':>13}{'top-10% precision':>19}")
        for name, m in t["models"].items():
            print(f"{name:<22}{m['test']['roc_auc']:>10}{m['recent']['roc_auc']:>12}"
                  f"{m['test']['pr_auc']:>13}{m['test']['precision_top_10pct']:>19}")
        print("lightgbm test AUC 95% CI:", t["models"]["lightgbm"]["test"]["roc_auc_ci95"])
        if any(m["test"]["roc_auc"] > 0.85 for m in t["models"].values()):
            print("!! a test AUC above 0.85 — investigate for leakage before believing it")
    if "by_sponsor_class" in report["targets"]["label"]:
        print("\nlabel AUC by sponsor class:", report["targets"]["label"]["by_sponsor_class"])
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_ml_train.py -v` → all pass. If the synthetic AUC threshold is flaky, strengthen the signal (larger `n` or smaller noise), never lower `> 0.85` below 0.8. Then `uv run pytest -q && uv run ruff check src tests`.

Note for the implementer: `test_ml_score.py`'s existing tests do not touch `train.py`; the registry's `save(...)` serializes the `dict[str, RiskModel]` with joblib (verified in Task 5's object design — `RiskModel` holds only sklearn/LightGBM objects).

---

### Task 7: Scoring both risks

**Files:** Modify `src/ctrisk/ml/score.py`, `README.md`

- [ ] **Step 1: Replace the `__main__` block** of `src/ctrisk/ml/score.py` (the two pure functions and their tests are unchanged):

```python
if __name__ == "__main__":
    from snowflake.connector.pandas_tools import write_pandas

    from ctrisk.config import load_config
    from ctrisk.ml.registry import MODELS_DIR, latest, load
    from ctrisk.warehouse.snowflake import connect

    load_config()
    version = latest(MODELS_DIR)
    models, _ = load(MODELS_DIR, version)
    overall, enrollment = models["label"], models["label_enrollment"]

    with connect() as conn:
        frame = conn.cursor().execute("SELECT * FROM TRIAL_FEATURES WHERE split = 'score'").fetch_pandas_all()
        frame.columns = frame.columns.str.lower()
        if frame.empty:
            raise SystemExit("no active trials to score (split = 'score' is empty)")
        p = overall.predict_proba(frame)
        drivers = top_drivers(overall.contributions(frame), overall.feature_names)
        scores = pd.DataFrame({
            "NCT_ID": frame["nct_id"], "MODEL_VERSION": f"v{version}",
            "RISK_SCORE": p.round(4), "RISK_DECILE": risk_deciles(p),
            "ENROLLMENT_RISK_SCORE": enrollment.predict_proba(frame).round(4),
            "TOP_DRIVER_1": [d[0] for d in drivers], "TOP_DRIVER_2": [d[1] for d in drivers],
            "TOP_DRIVER_3": [d[2] for d in drivers],
            "SCORED_AT": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")})
        ok, _, rows, _ = write_pandas(conn, scores, "TRIAL_RISK_SCORES", auto_create_table=True)
    print(f"appended {rows} scores from v{version} to TRIAL_RISK_SCORES" if ok else "write failed")
```

Update the module docstring: `"""Score active trials with the latest version: overall and enrollment termination risk, plus top drivers."""`

- [ ] **Step 2: README `## Run`** — add `make text` after `make attributes` with the comment `# -> data/parquet/trial_text`, and change the `make score` comment to `# overall + enrollment risk for active trials -> TRIAL_RISK_SCORES`.

- [ ] **Step 3: Run** — `uv run pytest -q && uv run ruff check src tests && uv run python -c "import ctrisk.ml.train, ctrisk.ml.score"` → all pass.

---

### Task 8: Real run (coordinator)

Cloud cost: GCS +~70 MB of text (≈$0.01 egress on load); Snowflake compute from trial credits.

- [ ] **Step 1: Rebuild Spark outputs**
  `make ingest-aact AACT=data/raw/aact_20260926.zip && make trials && make attributes && make text`
  Expected: 13 `extracted` lines; trials summary unchanged (`labeled 66124`); `{'trials': 96018}` twice; median text length ≈ 1,500–3,000 chars. Sanity-check `stop_reason` shares on real data (expect ≈ enrollment 33%, business 23%, safety 11%, efficacy 11%, other ≈ 12%, null ≈ 10%).
- [ ] **Step 2: Upload and build** — `make upload && make warehouse`. Expected: checks pass; the split table unchanged.
- [ ] **Step 3: With Frank's OK, delete the accidental v2** — `rm -r models/v2` and `DROP TABLE TRIAL_FEATURES_V2` — so the next version is v2 and the registry has no gap. (If Frank prefers to keep it, the next version is simply v3.)
- [ ] **Step 4: Train** — `make train` (expect 5–10 min: 12-point grid × 3 targets plus ablations). Record every printed number.
- [ ] **Step 5: Stop rule** — if any test AUC > 0.85, stop and investigate the top driver and the `no_text`/`no_burden` ablations before anything else.
- [ ] **Step 6: Compare with v1** — overall test AUC and CI, by-sponsor-class AUCs (the academic gap is the story), ablations (text lift, FAERS lift on `label_safety` = the fair Q3 answer), enrollment-model AUC.
- [ ] **Step 7: Score** — `make score`; in Snowflake: `SELECT RISK_DECILE, COUNT(*), ROUND(AVG(RISK_SCORE),3), ROUND(AVG(ENROLLMENT_RISK_SCORE),3) FROM TRIAL_RISK_SCORES WHERE MODEL_VERSION = 'v2' GROUP BY 1 ORDER BY 1`.
- [ ] **Step 8: Record** — README `## Results` (v1 → v2 with what changed each number), update the progress dashboard, tell Frank what the resume bullet can truthfully say.

## After M5

**M6 — point-in-time enrollment from archived AACT snapshots** (separate plan): verify in a browser how far back the monthly flat-file archives go, then for trials starting 2017–2018 read planned enrollment, planned duration, and site count from the snapshot of their start month. This is the one route to the strongest missing feature without leakage.

## Changes from review (applied before the real run)

Data side (real-data review of Tasks 1–3):
- **Stop-reason negation:** 52% of "safety" matches were disclaimers ("no safety concerns"). Negated clauses are now stripped before matching and patterns broadened; real shares: enrollment 31%, business 21%, efficacy 8%, safety 6%, other 24%, blank 10%.
- **Outcome leakage:** outcome measures are rewritten when results are posted (after start, label-correlated). Outcome counts and outcome text were removed.
- **Termination wording:** sentences like "the study was terminated…" are scrubbed from summaries before text features.
- **Sponsor activity:** "concurrent trials" counted only eventually-finished studies (skewed at scoring time); replaced by `sponsor_trials_started_2y`, which depends on start dates only.

Model side (review of Tasks 4–7):
- One `TextFeatures` per target, fit on its training rows and shared across ablations (identical results, ~2× faster); early stopping and selection both use AUC; `txt_*` contributions reported as one `registration_text` driver; scoring adds the new column to an existing `TRIAL_RISK_SCORES` and refuses pre-M5 model versions.
- New tests pin that text is only ever fit on training rows and that reason targets count and evaluate only their own rows.
