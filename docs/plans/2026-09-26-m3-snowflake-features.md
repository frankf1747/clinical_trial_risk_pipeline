# M3: GCS → Snowflake → Leakage-Safe Features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do not commit** — Frank commits this repo himself. **Do not create cloud resources** — Task 6 is done with Frank.

**Goal:** One `TRIAL_FEATURES` table in Snowflake — one row per trial, every feature computed only from what was known when the trial started — built from Parquet on GCS.

**Architecture:** Spark adds two small tables: per-trial attributes (countries, eligibility, disease area) and sponsor outcomes (every finished study with its end date). Parquet is synced to GCS; Snowflake loads it through an external stage into typed `RAW_*` tables, then portable SQL builds the time-aware features (FAERS history and sponsor track record strictly before each start date). The same feature SQL runs on DuckDB in tests, so leakage tests need no Snowflake.

**Tech Stack:** PySpark 3.5, GCS (`gcloud storage rsync`), Snowflake (key-pair auth, `snowflake-connector-python`), DuckDB (tests only), pytest.

**Budget:** GCS ≈ 100 MB ≈ $0.002/month; Snowflake reading it from AWS ≈ $0.01 per load. Snowflake compute comes from trial credits (X-Small warehouse, auto-suspend 60 s).

---

## Feature definitions (the contract)

| Column | Source | Point-in-time rule |
|---|---|---|
| `phase, number_of_arms, allocation, intervention_model, primary_purpose, masking, sponsor_class` | `RAW_TRIALS` | Latest record version (AACT keeps no history); low risk |
| `n_countries, us_only, min_age_years, max_age_years, healthy_volunteers, sex, criteria_count, criteria_chars, area_*` | `RAW_TRIAL_ATTRIBUTES` | Record fields (country caveat in spec) |
| `sponsor_prior_trials, sponsor_prior_termination_rate` | `RAW_SPONSOR_OUTCOMES` | Same lead sponsor, **completed before** this trial's `start_date`, excluding itself |
| `n_substances, faers_reports, faers_reports_12m, faers_serious_share, faers_death_share, has_faers_history` | `RAW_FAERS_DRUG_EVENTS` via monthly aggregates | Only report months **ending before** the trial's start month (conservative: the start month itself is excluded). `n_substances` counts only drugs with such history — a drug first reported after the start would reveal it later reached market |
| `label` | `RAW_TRIALS` | Outcome — the only post-start field |
| `split` | `start_date` | `train` 2008–2016, `test` 2017–2020, `score` = active (no label) |

Never in `TRIAL_FEATURES`: `status` (it *is* the label), `completion_date`, `sponsor_name`.

`start_date` is kept for splitting, not as a model input. The test split holds only trials finished by the 2026 snapshot, and terminated trials finish sooner, so recent years are enriched for terminations — M4 reports this alongside the AUC.

FAERS counts come from the Q1-per-year sample, so they are ~¼ scale; relative differences between drugs are what the model uses.

## File map

| File | Responsibility |
|---|---|
| `src/ctrisk/spark/clean_trials.py` | + `build_sponsor_outcomes` |
| `src/ctrisk/spark/trial_attributes.py` | Countries, eligibility, disease area per trial |
| `src/ctrisk/ingest/aact.py` | + `countries`, `eligibilities`, `browse_conditions` |
| `src/ctrisk/warehouse/sql.py` | Split and run SQL files; run named checks (any DB-API connection) |
| `src/ctrisk/warehouse/snowflake.py` | Connect, load from stage, build features, run checks |
| `sql/00_setup.sql` | One-time Snowflake setup (Frank runs in Snowsight) |
| `sql/10_raw_tables.sql` | Typed `RAW_*` tables (portable) |
| `sql/11_copy.sql` | `COPY INTO` from the GCS stage (Snowflake only) |
| `sql/20_features/*.sql` | INT tables and `TRIAL_FEATURES` (portable) |
| `sql/90_checks.sql` | Queries that must return zero rows (portable) |
| `tests/test_sponsor_outcomes.py`, `tests/test_trial_attributes.py`, `tests/test_warehouse_sql.py`, `tests/test_features_sql.py` | Tests |
| `tests/fixtures/aact/{countries,eligibilities,browse_conditions}.txt`, `studies.txt` (+ `completion_date`) | Fixtures |
| `Makefile`, `.env.example`, `pyproject.toml`, `README.md` | Wiring |

---

### Task 1: Sponsor outcomes

**Files:** Modify `tests/fixtures/aact/studies.txt`, `src/ctrisk/spark/clean_trials.py`; create `tests/test_sponsor_outcomes.py`

- [ ] **Step 1: Add `completion_date` to the studies fixture** — replace `tests/fixtures/aact/studies.txt` with:

```
nct_id|study_type|overall_status|phase|start_date|number_of_arms|completion_date
NCT001|INTERVENTIONAL|COMPLETED|PHASE2|2012-03-01|2|2014-01-01
NCT002|INTERVENTIONAL|TERMINATED|PHASE3|2015-06-15|2|2016-01-01
NCT003|INTERVENTIONAL|Active, not recruiting|PHASE1|2024-01-10|1|
NCT004|OBSERVATIONAL|COMPLETED|NA|2012-01-01||2013-01-01
NCT005|INTERVENTIONAL|COMPLETED|PHASE4|2012-01-01|2|2013-01-01
NCT006|INTERVENTIONAL|TERMINATED|PHASE2|2005-01-01|2|2006-01-01
NCT007|INTERVENTIONAL|WITHDRAWN|PHASE2|2012-01-01|2|
NCT008|INTERVENTIONAL|COMPLETED|PHASE1/PHASE2|2019-11-30|1|2020-06-01
NCT009|Interventional|Terminated|Phase 1/Phase 2|2018-05-01|3|2019-01-01
NCT010|INTERVENTIONAL|COMPLETED|PHASE2|2021-02-01|2|2022-01-01
```

(Only the new column changes; NCT003 already uses the legacy `Active, not recruiting` spelling.)

- [ ] **Step 2: Write failing test** — `tests/test_sponsor_outcomes.py`

```python
from ctrisk.spark.clean_trials import build_sponsor_outcomes


def test_every_finished_interventional_study_with_a_lead_sponsor(aact):
    rows = build_sponsor_outcomes(aact["studies"], aact["sponsors"]).collect()
    got = {r.nct_id: (r.sponsor_name, r.terminated, str(r.completion_date)) for r in rows}
    # Any phase counts as sponsor history; studies without a lead sponsor, observational,
    # withdrawn, and unfinished studies are excluded.
    assert got == {
        "NCT001": ("Merck", 0, "2014-01-01"),
        "NCT002": ("State University", 1, "2016-01-01"),
        "NCT009": ("Pfizer", 1, "2019-01-01"),
    }
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_sponsor_outcomes.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_sponsor_outcomes'`

- [ ] **Step 4: Implement** — add to `src/ctrisk/spark/clean_trials.py` after `build_drug_interventions`:

```python
def build_sponsor_outcomes(studies: DataFrame, sponsors: DataFrame) -> DataFrame:
    """Every finished interventional study, with its lead sponsor and end date.

    This is the history sponsor features look back on; Snowflake only uses rows that
    ended before a given trial started.
    """
    status = norm(F.col("overall_status"))
    lead = (sponsors.where(F.lower("lead_or_collaborator") == "lead")
            .select("nct_id", F.col("name").alias("sponsor_name"))
            .dropDuplicates(["nct_id"]))
    return (studies
            .where((norm(F.col("study_type")) == "INTERVENTIONAL")
                   & status.isin("COMPLETED", "TERMINATED")
                   & F.col("completion_date").isNotNull())
            .select("nct_id",
                    (status == "TERMINATED").cast("int").alias("terminated"),
                    F.to_date("completion_date").alias("completion_date"))
            .join(lead, "nct_id"))
```

In the `__main__` block, after the `drug_interventions` write, add:

```python
    build_sponsor_outcomes(t["studies"], t["sponsors"]) \
        .write.mode("overwrite").parquet(cfg.path("parquet", "sponsor_outcomes"))
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass; ruff clean.

---

### Task 2: Trial attributes

**Files:** Modify `src/ctrisk/ingest/aact.py`, `tests/conftest.py`; create `tests/fixtures/aact/{countries,eligibilities,browse_conditions}.txt`, `src/ctrisk/spark/trial_attributes.py`, `tests/test_trial_attributes.py`

- [ ] **Step 1: Extend AACT tables** — in `src/ctrisk/ingest/aact.py`:

```python
TABLES = ("studies", "designs", "sponsors", "interventions",
          "intervention_other_names", "browse_interventions",
          "countries", "eligibilities", "browse_conditions")
```

In `tests/conftest.py`, extend the `aact` fixture tuple the same way (all nine names).

- [ ] **Step 2: Write fixtures**

`tests/fixtures/aact/countries.txt`
```
id|nct_id|name|removed
1|NCT001|United States|f
2|NCT001|Canada|t
3|NCT002|United States|f
```

`tests/fixtures/aact/eligibilities.txt`
```
id|nct_id|gender|minimum_age|maximum_age|healthy_volunteers|criteria
1|NCT001|ALL|18 Years|75 Years|f|Inclusion Criteria:~1. Adults~2. Measurable disease~Exclusion Criteria:~1. Pregnancy
2|NCT003|FEMALE|6 Months||t|
```

`tests/fixtures/aact/browse_conditions.txt`
```
id|nct_id|mesh_term|downcase_mesh_term|mesh_type
1|NCT001|Lung Neoplasms|lung neoplasms|mesh-list
2|NCT001|Neoplasms|neoplasms|mesh-ancestor
3|NCT001|Respiratory Tract Diseases|respiratory tract diseases|mesh-ancestor
4|NCT002|Arthritis, Rheumatoid|arthritis, rheumatoid|mesh-list
5|NCT002|Immune System Diseases|immune system diseases|mesh-ancestor
```

- [ ] **Step 3: Write failing tests** — `tests/test_trial_attributes.py`

```python
import pytest

from ctrisk.spark.clean_trials import build_trials
from ctrisk.spark.trial_attributes import AREAS, build_trial_attributes


@pytest.fixture(scope="module")
def attrs(aact):
    trials = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])
    df = build_trial_attributes(trials, aact["countries"], aact["eligibilities"], aact["browse_conditions"])
    return {r.nct_id: r for r in df.collect()}


def test_one_row_per_trial(attrs):
    assert set(attrs) == {"NCT001", "NCT002", "NCT003", "NCT009"}


def test_countries_include_removed_ones(attrs):
    assert (attrs["NCT001"].n_countries, attrs["NCT001"].us_only) == (2, False)
    assert (attrs["NCT002"].n_countries, attrs["NCT002"].us_only) == (1, True)
    assert (attrs["NCT009"].n_countries, attrs["NCT009"].us_only) == (0, False)


def test_eligibility(attrs):
    a = attrs["NCT001"]
    assert (a.min_age_years, a.max_age_years, a.healthy_volunteers, a.sex) == (18.0, 75.0, False, "ALL")
    assert a.criteria_count == 3            # headers are not criteria
    b = attrs["NCT003"]
    assert (b.min_age_years, b.max_age_years, b.healthy_volunteers, b.criteria_count) == (0.5, None, True, 0)


def test_disease_areas(attrs):
    assert attrs["NCT001"].area_neoplasms and attrs["NCT001"].area_respiratory
    assert attrs["NCT002"].area_immune and not attrs["NCT002"].area_neoplasms
    assert not any(attrs["NCT009"][f"area_{k}"] for k in AREAS)
```

- [ ] **Step 4: Run to verify failure**

Run: `uv run pytest tests/test_trial_attributes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ctrisk.spark.trial_attributes'`

- [ ] **Step 5: Implement** — `src/ctrisk/spark/trial_attributes.py`

```python
"""Per-trial attributes from the registration record: geography, eligibility, disease area.

Countries include ones later removed from the record, so the count is every country the
trial ever listed. Record edits can still shift it; M4 measures the effect.
"""
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from ctrisk.config import load_config
from ctrisk.spark.clean_trials import read_table
from ctrisk.spark.session import get_spark

# Top-level MeSH disease categories, as AACT lists them among each trial's condition ancestors
AREAS = {
    "neoplasms": "Neoplasms",
    "cardiovascular": "Cardiovascular Diseases",
    "nervous_system": "Nervous System Diseases",
    "mental": "Mental Disorders",
    "infections": "Infections",
    "respiratory": "Respiratory Tract Diseases",
    "digestive": "Digestive System Diseases",
    "metabolic": "Nutritional and Metabolic Diseases",
    "immune": "Immune System Diseases",
    "skin": "Skin and Connective Tissue Diseases",
    "musculoskeletal": "Musculoskeletal Diseases",
    "urogenital": "Urogenital Diseases",
    "blood": "Hemic and Lymphatic Diseases",
    "endocrine": "Endocrine System Diseases",
}


def age_years(col: Column) -> Column:
    """'18 Years' -> 18.0, '6 Months' -> 0.5; anything else -> null."""
    n = F.regexp_extract(col, r"(\d+(?:\.\d+)?)", 1).cast("double")
    unit = F.lower(F.regexp_extract(col, r"(?i)(year|month|week|day)", 1))
    per_year = F.when(unit == "year", 1).when(unit == "month", 12).when(unit == "week", 52) \
        .when(unit == "day", 365)
    return F.round(n / per_year, 2)


def build_trial_attributes(trials: DataFrame, countries: DataFrame, eligibilities: DataFrame,
                           browse_conditions: DataFrame) -> DataFrame:
    geo = countries.groupBy("nct_id").agg(
        F.countDistinct("name").alias("n_countries"),
        F.expr("bool_and(name = 'United States')").alias("us_only"))

    # AACT stores line breaks in criteria as '~'; each non-empty, non-header line is one criterion
    lines = F.split(F.coalesce(F.col("criteria"), F.lit("")), "~")
    is_criterion = lambda x: (F.trim(x) != "") & ~F.lower(F.trim(x)).endswith("criteria:")  # noqa: E731
    elig = eligibilities.select(
        "nct_id",
        age_years(F.col("minimum_age")).alias("min_age_years"),
        age_years(F.col("maximum_age")).alias("max_age_years"),
        (F.col("healthy_volunteers") == "t").alias("healthy_volunteers"),
        F.upper("gender").alias("sex"),
        F.size(F.filter(lines, is_criterion)).alias("criteria_count"),
        F.length(F.coalesce(F.col("criteria"), F.lit(""))).alias("criteria_chars"))

    area = (browse_conditions
            .where(F.col("mesh_term").isin(list(AREAS.values())))
            .groupBy("nct_id")
            .agg(*[F.max(F.col("mesh_term") == term).alias(f"area_{key}") for key, term in AREAS.items()]))

    return (trials.select("nct_id")
            .join(geo, "nct_id", "left")
            .join(elig, "nct_id", "left")
            .join(area, "nct_id", "left")
            .fillna(0, subset=["n_countries"])
            .fillna(False, subset=["us_only", *[f"area_{k}" for k in AREAS]]))


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("trial_attributes", cfg.mode)
    aact = cfg.path("raw", "aact")
    attrs = build_trial_attributes(spark.read.parquet(cfg.path("parquet", "trials")),
                                   read_table(spark, aact, "countries"),
                                   read_table(spark, aact, "eligibilities"),
                                   read_table(spark, aact, "browse_conditions"))
    attrs.write.mode("overwrite").parquet(cfg.path("parquet", "trial_attributes"))
    print({"trials": attrs.count()})
```

If ruff rejects the `lambda` assignment even with `noqa`, turn `is_criterion` into a small module-level `def`.

- [ ] **Step 6: Run all tests**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass; ruff clean.

---

### Task 3: SQL runner

**Files:** Create `src/ctrisk/warehouse/__init__.py` (empty), `src/ctrisk/warehouse/sql.py`, `tests/test_warehouse_sql.py`; modify `pyproject.toml`

- [ ] **Step 1: Add dependencies**

Run: `uv add snowflake-connector-python && uv add --dev duckdb`

- [ ] **Step 2: Write failing tests** — `tests/test_warehouse_sql.py`

```python
import duckdb

from ctrisk.warehouse.sql import failed_checks, statements


def test_splits_on_line_ending_semicolons_and_skips_comment_only_chunks(tmp_path):
    f = tmp_path / "a.sql"
    f.write_text("-- header\nCREATE TABLE t (x INT);\n\nINSERT INTO t VALUES (1);\n-- trailing note\n")
    assert statements(f) == ["-- header\nCREATE TABLE t (x INT)", "INSERT INTO t VALUES (1)"]


def test_named_checks_report_rows_that_break_a_rule(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1), (1), (2)) v(x)")
    f = tmp_path / "checks.sql"
    f.write_text("-- check: x is unique\nSELECT x FROM t GROUP BY x HAVING COUNT(*) > 1;\n"
                 "-- check: x is positive\nSELECT x FROM t WHERE x <= 0;\n")
    assert failed_checks(con, f) == ["x is unique: 1 rows, e.g. [(1,)]"]
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_warehouse_sql.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ctrisk.warehouse'`

- [ ] **Step 4: Implement** — `src/ctrisk/warehouse/sql.py`

```python
"""Run SQL files on any DB-API connection: Snowflake in the pipeline, DuckDB in tests."""
import re
from pathlib import Path

SQL_DIR = Path(__file__).resolve().parents[3] / "sql"


def statements(path: Path) -> list[str]:
    """Split on semicolons that end a line; drop chunks that are only comments."""
    chunks = re.split(r";\s*$", path.read_text(), flags=re.MULTILINE)
    return [c.strip() for c in chunks
            if any(line.strip() and not line.strip().startswith("--") for line in c.splitlines())]


def run_files(cursor, paths: list[Path]) -> None:
    for path in paths:
        for statement in statements(path):
            cursor.execute(statement)


def failed_checks(cursor, path: Path) -> list[str]:
    """Each check is a query returning the rows that break a rule, named by a '-- check:' line."""
    failures = []
    for statement in statements(path):
        name = re.search(r"--\s*check:\s*(.+)", statement).group(1).strip()
        cursor.execute(statement)
        rows = cursor.fetchall()
        if rows:
            failures.append(f"{name}: {len(rows)} rows, e.g. {rows[:3]}")
    return failures
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass; ruff clean.

---

### Task 4: Feature SQL and leakage tests

**Files:** Create `sql/10_raw_tables.sql`, `sql/20_features/10_int_faers_monthly.sql`, `sql/20_features/20_int_drug_safety.sql`, `sql/20_features/30_int_sponsor_history.sql`, `sql/20_features/40_trial_features.sql`, `sql/90_checks.sql`, `tests/test_features_sql.py`

All SQL in this task must run unchanged on Snowflake and DuckDB: use `CASE` (not `IFF`), `NULLIF` for division, `INTERVAL '12 months'`, `COUNT_IF`, `CREATE OR REPLACE TABLE ... AS`.

- [ ] **Step 1: Write `sql/10_raw_tables.sql`** — column names match the Parquet written by Spark.

```sql
-- Typed landing tables. Snowflake fills them with COPY (11_copy.sql); tests fill them with INSERTs.

CREATE OR REPLACE TABLE RAW_TRIALS (
    nct_id STRING, status STRING, phase STRING, start_date DATE, number_of_arms INT, label INT,
    allocation STRING, intervention_model STRING, primary_purpose STRING, masking STRING,
    sponsor_name STRING, sponsor_class STRING
);

CREATE OR REPLACE TABLE RAW_TRIAL_ATTRIBUTES (
    nct_id STRING, n_countries INT, us_only BOOLEAN, min_age_years FLOAT, max_age_years FLOAT,
    healthy_volunteers BOOLEAN, sex STRING, criteria_count INT, criteria_chars INT,
    area_neoplasms BOOLEAN, area_cardiovascular BOOLEAN, area_nervous_system BOOLEAN,
    area_mental BOOLEAN, area_infections BOOLEAN, area_respiratory BOOLEAN, area_digestive BOOLEAN,
    area_metabolic BOOLEAN, area_immune BOOLEAN, area_skin BOOLEAN, area_musculoskeletal BOOLEAN,
    area_urogenital BOOLEAN, area_blood BOOLEAN, area_endocrine BOOLEAN
);

CREATE OR REPLACE TABLE RAW_SPONSOR_OUTCOMES (
    nct_id STRING, terminated INT, completion_date DATE, sponsor_name STRING
);

CREATE OR REPLACE TABLE RAW_TRIAL_DRUG_MAP (
    nct_id STRING, substance STRING
);

CREATE OR REPLACE TABLE RAW_FAERS_DRUG_EVENTS (
    safetyreportid STRING, substance STRING, receivedate DATE, receiptdate DATE,
    serious BOOLEAN, death BOOLEAN, suspect BOOLEAN, harmonized BOOLEAN, active_substance BOOLEAN
);
```

- [ ] **Step 2: Write the feature SQL**

`sql/20_features/10_int_faers_monthly.sql`
```sql
-- Reports per substance per month: 12M report rows -> ~0.5M, so the per-trial join stays small.
CREATE OR REPLACE TABLE INT_FAERS_MONTHLY AS
SELECT substance,
       DATE_TRUNC('month', receivedate) AS month,
       COUNT(*)            AS reports,
       COUNT_IF(serious)   AS serious_reports,
       COUNT_IF(death)     AS death_reports
FROM RAW_FAERS_DRUG_EVENTS
WHERE receivedate IS NOT NULL
GROUP BY substance, DATE_TRUNC('month', receivedate);
```

`sql/20_features/20_int_drug_safety.sql`
```sql
-- FAERS history of each trial's drugs, counting only months that ended before the trial's
-- start month. The start month itself is excluded, so no report dated on or after the start
-- can leak in. A report naming two of a trial's drugs counts once per drug.
CREATE OR REPLACE TABLE INT_DRUG_SAFETY AS
SELECT t.nct_id,
       COUNT(DISTINCT m.substance)            AS n_substances,
       COALESCE(SUM(f.reports), 0)            AS faers_reports,
       COALESCE(SUM(f.serious_reports), 0)    AS faers_serious_reports,
       COALESCE(SUM(f.death_reports), 0)      AS faers_death_reports,
       COALESCE(SUM(CASE WHEN f.month >= DATE_TRUNC('month', t.start_date) - INTERVAL '12 months'
                         THEN f.reports END), 0) AS faers_reports_12m
FROM RAW_TRIALS t
JOIN RAW_TRIAL_DRUG_MAP m ON m.nct_id = t.nct_id
LEFT JOIN INT_FAERS_MONTHLY f
       ON f.substance = m.substance
      AND f.month < DATE_TRUNC('month', t.start_date)
GROUP BY t.nct_id;
```

`sql/20_features/30_int_sponsor_history.sql`
```sql
-- The lead sponsor's track record: its studies that finished before this trial started.
CREATE OR REPLACE TABLE INT_SPONSOR_HISTORY AS
SELECT t.nct_id,
       COUNT(s.nct_id)                    AS sponsor_prior_trials,
       AVG(s.terminated::FLOAT)           AS sponsor_prior_termination_rate
FROM RAW_TRIALS t
LEFT JOIN RAW_SPONSOR_OUTCOMES s
       ON s.sponsor_name = t.sponsor_name
      AND s.completion_date < t.start_date
      AND s.nct_id <> t.nct_id
GROUP BY t.nct_id;
```

`sql/20_features/40_trial_features.sql`
```sql
-- One row per trial. Everything except `label` was knowable on the trial's start date.
-- Deliberately absent: status (it is the label), completion_date, sponsor_name.
CREATE OR REPLACE TABLE TRIAL_FEATURES AS
SELECT t.nct_id,
       t.label,
       CASE WHEN t.label IS NULL THEN 'score'
            WHEN t.start_date < DATE '2017-01-01' THEN 'train'
            ELSE 'test' END                                        AS split,
       t.start_date,
       t.phase, t.number_of_arms, t.allocation, t.intervention_model, t.primary_purpose,
       t.masking, t.sponsor_class,
       a.* EXCLUDE (nct_id),
       COALESCE(sh.sponsor_prior_trials, 0)                        AS sponsor_prior_trials,
       sh.sponsor_prior_termination_rate,
       COALESCE(ds.n_substances, 0)                                AS n_substances,
       COALESCE(ds.faers_reports, 0)                               AS faers_reports,
       COALESCE(ds.faers_reports_12m, 0)                           AS faers_reports_12m,
       ds.faers_serious_reports::FLOAT / NULLIF(ds.faers_reports, 0) AS faers_serious_share,
       ds.faers_death_reports::FLOAT / NULLIF(ds.faers_reports, 0)   AS faers_death_share,
       COALESCE(ds.faers_reports, 0) > 0                           AS has_faers_history
FROM RAW_TRIALS t
LEFT JOIN RAW_TRIAL_ATTRIBUTES a ON a.nct_id = t.nct_id
LEFT JOIN INT_SPONSOR_HISTORY sh ON sh.nct_id = t.nct_id
LEFT JOIN INT_DRUG_SAFETY ds     ON ds.nct_id = t.nct_id;
```

`sql/90_checks.sql`
```sql
-- check: one row per trial
SELECT nct_id FROM TRIAL_FEATURES GROUP BY nct_id HAVING COUNT(*) > 1;

-- check: every trial has a feature row
SELECT nct_id FROM RAW_TRIALS WHERE nct_id NOT IN (SELECT nct_id FROM TRIAL_FEATURES);

-- check: labels only in train and test
SELECT nct_id FROM TRIAL_FEATURES WHERE (label IS NULL) <> (split = 'score');

-- check: test split is not empty
SELECT COUNT(*) FROM TRIAL_FEATURES WHERE split = 'test' HAVING COUNT(*) = 0;

-- check: shares are proportions
SELECT nct_id FROM TRIAL_FEATURES
WHERE faers_serious_share NOT BETWEEN 0 AND 1 OR faers_death_share NOT BETWEEN 0 AND 1;
```

- [ ] **Step 3: Write failing tests** — `tests/test_features_sql.py`

```python
"""Run the real feature SQL on DuckDB with hand-built rows, including leakage traps."""
import duckdb
import pytest

from ctrisk.warehouse.sql import SQL_DIR, failed_checks, run_files

FEATURE_FILES = sorted((SQL_DIR / "20_features").glob("*.sql"))


@pytest.fixture(scope="module")
def db():
    con = duckdb.connect()
    run_files(con, [SQL_DIR / "10_raw_tables.sql"])
    con.execute("""INSERT INTO RAW_TRIALS (nct_id, start_date, label, sponsor_name, sponsor_class) VALUES
        ('T1', DATE '2015-06-15', 1,    'Acme', 'INDUSTRY'),
        ('T2', DATE '2019-03-10', 0,    'Acme', 'INDUSTRY'),
        ('T3', DATE '2024-01-01', NULL, 'Beta', 'OTHER')""")
    con.execute("INSERT INTO RAW_TRIAL_ATTRIBUTES (nct_id, n_countries) VALUES ('T1', 3), ('T2', 1), ('T3', 1)")
    con.execute("INSERT INTO RAW_TRIAL_DRUG_MAP VALUES ('T1', 'drugx'), ('T2', 'drugx')")
    con.execute("""INSERT INTO RAW_FAERS_DRUG_EVENTS (safetyreportid, substance, receivedate, serious, death) VALUES
        ('r4', 'drugx', DATE '2014-01-10', false, true),   -- before T1, outside its 12-month window
        ('r1', 'drugx', DATE '2015-05-20', true,  false),  -- month before T1 starts: counted
        ('r2', 'drugx', DATE '2015-06-01', false, false),  -- T1's start month: excluded (conservative)
        ('r3', 'drugx', DATE '2015-06-16', true,  true)    -- day after T1 starts: must not count""")
    con.execute("""INSERT INTO RAW_SPONSOR_OUTCOMES VALUES
        ('S1', 1, DATE '2014-12-31', 'Acme'),   -- ended before T1: counts for T1 and T2
        ('S2', 0, DATE '2015-06-16', 'Acme'),   -- ended the day after T1 started: T2 only
        ('T1', 1, DATE '2016-01-01', 'Acme')    -- T1 itself: never its own history""")
    run_files(con, FEATURE_FILES)
    return con


def features(db, nct_id):
    cur = db.execute("SELECT * FROM TRIAL_FEATURES WHERE nct_id = ?", [nct_id])
    return dict(zip([c[0] for c in cur.description], cur.fetchone()))


def test_faers_counts_only_reports_before_the_start_month(db):
    t1 = features(db, "T1")
    assert t1["faers_reports"] == 2               # r4, r1 — not r2 (start month), not r3 (after start)
    assert t1["faers_reports_12m"] == 1           # r1 only
    assert t1["faers_serious_share"] == 0.5
    assert t1["faers_death_share"] == 0.5
    assert t1["has_faers_history"] is True
    assert features(db, "T2")["faers_reports"] == 4


def test_trial_without_matched_drugs_has_no_history(db):
    t3 = features(db, "T3")
    assert (t3["n_substances"], t3["faers_reports"], t3["has_faers_history"]) == (0, 0, False)
    assert t3["faers_serious_share"] is None


def test_sponsor_history_uses_only_trials_that_ended_before_start(db):
    t1, t2 = features(db, "T1"), features(db, "T2")
    assert (t1["sponsor_prior_trials"], t1["sponsor_prior_termination_rate"]) == (1, 1.0)
    assert t2["sponsor_prior_trials"] == 3
    assert t2["sponsor_prior_termination_rate"] == pytest.approx(2 / 3)
    assert features(db, "T3")["sponsor_prior_trials"] == 0


def test_split_by_start_year(db):
    assert [features(db, t)["split"] for t in ("T1", "T2", "T3")] == ["train", "test", "score"]


def test_post_start_fields_never_reach_features(db):
    cols = {c[0] for c in db.execute("SELECT * FROM TRIAL_FEATURES LIMIT 0").description}
    assert not cols & {"status", "completion_date", "sponsor_name"}


def test_checks_pass(db):
    assert failed_checks(db, SQL_DIR / "90_checks.sql") == []
```

- [ ] **Step 4: Run to verify failure, then pass**

The SQL files and tests are written in the same task, so first run the tests with `sql/20_features/` temporarily empty to see them fail (`CatalogException: Table ... TRIAL_FEATURES does not exist`), then restore the files and run:

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass. If DuckDB rejects syntax, adjust to a form both engines accept and note it — do not add engine-specific branches.

---

### Task 5: Snowflake loader and wiring

**Files:** Create `sql/00_setup.sql`, `sql/11_copy.sql`, `src/ctrisk/warehouse/snowflake.py`; modify `Makefile`, `.env.example`, `README.md`

- [ ] **Step 1: Write `sql/00_setup.sql`** (Frank runs this once in Snowsight)

```sql
-- One-time setup. Run in Snowsight as ACCOUNTADMIN, top to bottom.
-- Replace clinical-trial-risk-frankfu with your bucket name if different.

USE ROLE ACCOUNTADMIN;
CREATE STORAGE INTEGRATION IF NOT EXISTS CTRISK_GCS
    TYPE = EXTERNAL_STAGE
    STORAGE_PROVIDER = 'GCS'
    ENABLED = TRUE
    STORAGE_ALLOWED_LOCATIONS = ('gcs://clinical-trial-risk-frankfu/parquet/');
GRANT USAGE ON INTEGRATION CTRISK_GCS TO ROLE SYSADMIN;
SET me = CURRENT_USER();
GRANT ROLE SYSADMIN TO USER IDENTIFIER($me);   -- the pipeline connects as SYSADMIN
-- Copy STORAGE_GCP_SERVICE_ACCOUNT from this output for the bucket permission step:
DESC STORAGE INTEGRATION CTRISK_GCS;

-- Everything the pipeline owns belongs to SYSADMIN, not ACCOUNTADMIN.
USE ROLE SYSADMIN;
CREATE WAREHOUSE IF NOT EXISTS CTRISK_WH
    WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE;
CREATE DATABASE IF NOT EXISTS CTRISK;
CREATE SCHEMA IF NOT EXISTS CTRISK.PIPELINE;
CREATE FILE FORMAT IF NOT EXISTS CTRISK.PIPELINE.PARQUET_FF TYPE = PARQUET;
CREATE STAGE IF NOT EXISTS CTRISK.PIPELINE.PARQUET_STAGE
    URL = 'gcs://clinical-trial-risk-frankfu/parquet/'
    STORAGE_INTEGRATION = CTRISK_GCS
    FILE_FORMAT = CTRISK.PIPELINE.PARQUET_FF;
```

- [ ] **Step 2: Write `sql/11_copy.sql`** (Snowflake only)

```sql
-- Load each Parquet folder from the GCS stage. Tables were just recreated, so every file loads.
COPY INTO RAW_TRIALS FROM @PARQUET_STAGE/trials/
    PATTERN = '.*[.]parquet' MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO RAW_TRIAL_ATTRIBUTES FROM @PARQUET_STAGE/trial_attributes/
    PATTERN = '.*[.]parquet' MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO RAW_SPONSOR_OUTCOMES FROM @PARQUET_STAGE/sponsor_outcomes/
    PATTERN = '.*[.]parquet' MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO RAW_TRIAL_DRUG_MAP FROM @PARQUET_STAGE/trial_drug_map/
    PATTERN = '.*[.]parquet' MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO RAW_FAERS_DRUG_EVENTS FROM @PARQUET_STAGE/faers_drug_events/
    PATTERN = '.*[.]parquet' MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
```

- [ ] **Step 3: Write `src/ctrisk/warehouse/snowflake.py`**

```python
"""Load Parquet from the GCS stage into Snowflake, build TRIAL_FEATURES, and check it."""
import os

import snowflake.connector

from ctrisk.config import load_config
from ctrisk.gates import DataGateError
from ctrisk.warehouse.sql import SQL_DIR, failed_checks, run_files

BUILD = [SQL_DIR / "10_raw_tables.sql", SQL_DIR / "11_copy.sql",
         *sorted((SQL_DIR / "20_features").glob("*.sql"))]


def connect():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        private_key_file=os.path.expanduser(os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"]),
        role=os.getenv("SNOWFLAKE_ROLE", "SYSADMIN"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "CTRISK_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "CTRISK"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PIPELINE"),
    )


if __name__ == "__main__":
    cfg = load_config()  # also loads .env
    with connect() as conn, conn.cursor() as cur:
        cur.execute("LIST @PARQUET_STAGE/trials/")
        if not cur.fetchall():
            raise DataGateError("stage has no trials Parquet; run `make upload` first")
        run_files(cur, BUILD)
        if failures := failed_checks(cur, SQL_DIR / "90_checks.sql"):
            raise DataGateError("; ".join(failures))
        labeled = cur.execute("SELECT COUNT(label) FROM TRIAL_FEATURES").fetchone()[0]
        if labeled < cfg.min_trials:
            raise DataGateError(f"{labeled} labeled trials in TRIAL_FEATURES, expected {cfg.min_trials}+")
        cur.execute("""SELECT split, COUNT(*), ROUND(AVG(label), 3), ROUND(AVG(has_faers_history::INT), 3)
                       FROM TRIAL_FEATURES GROUP BY split ORDER BY split""")
        print("split | trials | termination rate | share with FAERS history")
        for row in cur.fetchall():
            print(" | ".join(str(v) for v in row))
```

- [ ] **Step 4: Update `Makefile`** — add `-include .env` as the first line, add `attributes upload warehouse` to `.PHONY`, and append:

```make
attributes:
	uv run python -m ctrisk.spark.trial_attributes

# Sync Spark output to GCS (Parquet only; ~100 MB)
upload:
	gcloud storage rsync --recursive --exclude='.*\.crc$$|.*_SUCCESS$$' data/parquet gs://$(GCP_BUCKET)/parquet

warehouse:
	uv run python -m ctrisk.warehouse.snowflake
```

- [ ] **Step 5: Update `.env.example`** — replace the `# Cloud (M5)` block with:

```bash
# GCS bucket for Parquet (M3) and Dataproc (M5)
GCP_BUCKET=clinical-trial-risk-frankfu

# Snowflake (key-pair auth; see README)
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_USER=
SNOWFLAKE_PRIVATE_KEY_FILE=~/.snowflake/rsa_key.p8
SNOWFLAKE_ROLE=SYSADMIN
SNOWFLAKE_WAREHOUSE=CTRISK_WH
SNOWFLAKE_DATABASE=CTRISK
SNOWFLAKE_SCHEMA=PIPELINE
```

- [ ] **Step 6: Update the README `## Run` block** — after `make match`, add:

```bash
make attributes                               # -> data/parquet/trial_attributes
make upload                                   # data/parquet -> gs://$GCP_BUCKET/parquet
make warehouse                                # Snowflake: RAW_* -> TRIAL_FEATURES, then checks
```

and add a `## Snowflake setup (once)` section: create key pair (Task 6 Step 3 commands), run `sql/00_setup.sql` in Snowsight, grant the integration's service account `roles/storage.objectViewer` on the bucket.

- [ ] **Step 7: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass; ruff clean.

---

### Task 6: Cloud setup (Frank + coordinator, one time)

Every command that creates something states its cost. Frank runs anything touching credentials or permissions.

- [ ] **Step 1 (coordinator, cost ≈ $0): create the bucket** — after Frank approves:

```bash
gcloud storage buckets create gs://clinical-trial-risk-frankfu --location=us-east1 \
  --uniform-bucket-level-access --public-access-prevention
```

- [ ] **Step 2 (Frank): run `sql/00_setup.sql` in Snowsight.** Copy `STORAGE_GCP_SERVICE_ACCOUNT` from the `DESC` output.

- [ ] **Step 3 (Frank): let Snowflake read the bucket**

```bash
gcloud storage buckets add-iam-policy-binding gs://clinical-trial-risk-frankfu \
  --member="serviceAccount:<STORAGE_GCP_SERVICE_ACCOUNT>" --role=roles/storage.objectViewer
```

- [ ] **Step 4 (Frank): key-pair login for the pipeline**

```bash
mkdir -p ~/.snowflake && openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out ~/.snowflake/rsa_key.p8 -nocrypt && chmod 600 ~/.snowflake/rsa_key.p8
openssl rsa -in ~/.snowflake/rsa_key.p8 -pubout | grep -v -- '-----' | tr -d '\n'
```

In Snowsight: `ALTER USER <your user> SET RSA_PUBLIC_KEY='<the printed key>';`

- [ ] **Step 5 (Frank): fill `.env`** with the Snowflake block from `.env.example`. `SNOWFLAKE_ACCOUNT` is the account identifier (Snowsight → account menu → Account → "Copy account identifier", e.g. `ORG-ACCOUNT`).

---

### Task 7: Run on real data

- [ ] **Step 1: Rebuild Spark outputs**

```bash
make ingest-aact AACT=data/raw/aact_20260926.zip && make trials && make attributes
```

Expected: 9 `extracted` lines; trials summary unchanged (`labeled 66124`); `{'trials': 96018}`. Then sanity-check the new tables in a short PySpark session: share of trials with ≥1 area flag (expect most), `min_age_years` null share (expect small), `sponsor_outcomes` row count (expect 350K+ finished interventional studies).

- [ ] **Step 2: Upload** (egress ≈ $0.01)

```bash
make upload && gcloud storage du -s gs://clinical-trial-risk-frankfu/parquet
```

Expected: ~100 MB.

- [ ] **Step 3: Build in Snowflake**

```bash
make warehouse
```

Expected: checks pass; three rows (`score`, `test`, `train`) with ~29.9K / ~?K / ~?K trials, termination rate near 15% for train/test, FAERS-history share ~0.6–0.7.

- [ ] **Step 4: Spot-check one real trial by hand** — pick a labeled trial with a common drug (e.g. one matched to `paclitaxel`), and in Snowsight compare its `faers_reports` with:

```sql
SELECT COUNT(*) FROM RAW_FAERS_DRUG_EVENTS e
JOIN RAW_TRIAL_DRUG_MAP m ON m.substance = e.substance
JOIN RAW_TRIALS t ON t.nct_id = m.nct_id
WHERE t.nct_id = '<id>' AND e.receivedate < DATE_TRUNC('month', t.start_date);
```

They must be equal.

- [ ] **Step 5: Record** — add an M3 block to README `## Status` (rows per split, termination rates, FAERS-history share, sponsor-history coverage), log progress, list changed files for Frank.
