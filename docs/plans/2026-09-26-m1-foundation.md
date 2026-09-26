# M1: Foundation + Trial Table — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an AACT snapshot into `trials` and `drug_interventions` Parquet tables — one row per eligible drug trial with its label — runnable locally with one command.

**Architecture:** `ingest/aact.py` extracts the four needed tables from an AACT zip (URL or local path). `spark/clean_trials.py` normalizes values, filters to the study population, attaches design and lead-sponsor fields, and writes Parquet. A data gate stops the run if the trial count or termination rate is implausible. Pure functions take DataFrames in and return DataFrames out, so tests run on tiny fixtures.

**Tech Stack:** Python 3.11, uv, PySpark 3.5 (matches Dataproc image 2.2), pytest, GitHub Actions.

**Spec:** `docs/specs/2026-09-26-design.md`

---

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies, package layout, pytest config |
| `Makefile` | `setup`, `test`, `ingest-aact`, `trials` |
| `.env.example` | Documented settings |
| `src/ctrisk/config.py` | Read `.env`; resolve local vs cloud paths |
| `src/ctrisk/gates.py` | `DataGateError` |
| `src/ctrisk/spark/session.py` | Build a SparkSession for the current mode |
| `src/ctrisk/ingest/aact.py` | Download/extract AACT tables |
| `src/ctrisk/spark/clean_trials.py` | Build `trials` and `drug_interventions` |
| `tests/fixtures/aact/*.txt` | 10 hand-checkable trials covering every filter rule |
| `tests/conftest.py` | Shared Spark fixture |
| `tests/test_aact_ingest.py`, `tests/test_clean_trials.py` | Tests |
| `.github/workflows/ci.yml` | Run tests on push |

---

### Task 1: Project skeleton

**Files:** Create `pyproject.toml`, `.gitignore`, `.env.example`, `Makefile`, `README.md`, `src/ctrisk/__init__.py`, `src/ctrisk/ingest/__init__.py`, `src/ctrisk/spark/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "ctrisk"
version = "0.1.0"
description = "Predict early termination of drug clinical trials"
requires-python = ">=3.11,<3.12"
dependencies = [
    "pyspark==3.5.3",
    "python-dotenv>=1.0",
    "requests>=2.32",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ctrisk"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
.DS_Store
.env
data/
models/*/model.pkl
spark-warehouse/
metastore_db/
derby.log
```

- [ ] **Step 3: Write `.env.example`**

```bash
# local = Spark on this machine, files under DATA_DIR
# cloud = Spark on Dataproc, files under gs://GCP_BUCKET (M5)
MODE=local
DATA_DIR=data

# Data gates
MIN_TRIALS=40000

# Cloud (M5)
GCP_BUCKET=
```

- [ ] **Step 4: Write `Makefile`**

```make
.PHONY: setup test ingest-aact trials

setup:
	uv sync

test:
	uv run pytest -q

# Usage: make ingest-aact AACT=<url-or-path-to-aact-zip>
ingest-aact:
	uv run python -m ctrisk.ingest.aact $(AACT)

trials:
	uv run python -m ctrisk.spark.clean_trials
```

- [ ] **Step 5: Write `README.md` stub and empty package files**

```markdown
# Clinical Trial Risk Pipeline

Predicts which drug trials will be terminated early, using only what is known when they start.

Work in progress — see `docs/specs/2026-09-26-design.md`.
```

Create empty `src/ctrisk/__init__.py`, `src/ctrisk/ingest/__init__.py`, `src/ctrisk/spark/__init__.py`.

- [ ] **Step 6: Install and verify Spark starts**

```bash
uv python pin 3.11
uv sync
uv run python -c "from pyspark.sql import SparkSession; print(SparkSession.builder.master('local[1]').getOrCreate().range(3).count())"
```

Expected: `3`. If Spark fails on the installed Java (18), install Java 17 and retry:

```bash
brew install openjdk@17
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
```

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "Add project skeleton"
```

---

### Task 2: Config, gates, Spark session

**Files:** Create `src/ctrisk/config.py`, `src/ctrisk/gates.py`, `src/ctrisk/spark/session.py`

These are thin glue; they are exercised by later tests rather than tested alone.

- [ ] **Step 1: Write `src/ctrisk/config.py`**

```python
"""Settings from .env. Everything that differs between local and cloud lives here."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    mode: str        # "local" or "cloud"
    local_root: str  # where ingest writes downloads
    data_root: str   # where Spark reads and writes
    min_trials: int

    def path(self, *parts: str) -> str:
        return "/".join([self.data_root, *parts])


def load_config() -> Config:
    load_dotenv()
    mode = os.getenv("MODE", "local")
    local_root = os.getenv("DATA_DIR", "data")
    if mode == "local":
        data_root = local_root
    elif mode == "cloud":
        data_root = f"gs://{os.environ['GCP_BUCKET']}"
    else:
        raise ValueError(f"MODE must be 'local' or 'cloud', got {mode!r}")
    return Config(mode, local_root, data_root, int(os.getenv("MIN_TRIALS", "40000")))
```

- [ ] **Step 2: Write `src/ctrisk/gates.py`**

```python
class DataGateError(RuntimeError):
    """A data sanity check failed. The pipeline stops rather than produce a misleading result."""
```

- [ ] **Step 3: Write `src/ctrisk/spark/session.py`**

```python
from pyspark.sql import SparkSession


def get_spark(app: str, mode: str = "local") -> SparkSession:
    builder = SparkSession.builder.appName(app).config("spark.sql.session.timeZone", "UTC")
    if mode == "local":
        builder = builder.master("local[*]")  # on Dataproc the cluster sets the master
    return builder.getOrCreate()
```

- [ ] **Step 4: Commit**

```bash
git add src/ctrisk
git commit -m "Add config, data gate error, Spark session"
```

---

### Task 3: AACT ingest

**Files:** Create `src/ctrisk/ingest/aact.py`, `tests/test_aact_ingest.py`

AACT publishes a daily zip of pipe-delimited `.txt` files, one per table (download page: https://aact.ctti-clinicaltrials.org/snapshots). The page blocks scripted requests, so ingest accepts either a URL or a zip you downloaded in a browser.

- [ ] **Step 1: Write the failing tests**

```python
import zipfile

import pytest

from ctrisk.ingest.aact import TABLES, fetch


def make_zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for name in names:
            z.writestr(f"export/{name}", "nct_id\nNCT001\n")


def test_extracts_only_needed_tables(tmp_path):
    zip_path = tmp_path / "aact.zip"
    make_zip(zip_path, [f"{t}.txt" for t in TABLES] + ["outcomes.txt"])

    out = fetch(str(zip_path), tmp_path / "aact")

    assert sorted(p.name for p in out) == sorted(f"{t}.txt" for t in TABLES)
    assert not (tmp_path / "aact" / "outcomes.txt").exists()


def test_missing_table_fails_loudly(tmp_path):
    zip_path = tmp_path / "aact.zip"
    make_zip(zip_path, ["studies.txt"])

    with pytest.raises(FileNotFoundError, match="designs.txt"):
        fetch(str(zip_path), tmp_path / "aact")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aact_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ctrisk.ingest.aact'`

- [ ] **Step 3: Implement `src/ctrisk/ingest/aact.py`**

```python
"""Extract the AACT tables this project uses from a snapshot zip (URL or local path)."""
import shutil
import sys
import zipfile
from pathlib import Path

import requests

from ctrisk.config import load_config

TABLES = ("studies", "designs", "sponsors", "interventions")


def fetch(source: str, dest: Path, tables: tuple[str, ...] = TABLES) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = _download(source, dest.parent / "aact.zip") if source.startswith("http") else Path(source)
    return _extract(zip_path, dest, tables)


def _download(url: str, target: Path) -> Path:
    if target.exists():
        print(f"skip download, {target} exists")
        return target
    partial = target.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(partial, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    partial.rename(target)
    return target


def _extract(zip_path: Path, dest: Path, tables: tuple[str, ...]) -> list[Path]:
    wanted = {f"{t}.txt" for t in tables}
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            name = Path(member).name
            if name in wanted:
                with z.open(member) as src, open(dest / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                out.append(dest / name)
    missing = wanted - {p.name for p in out}
    if missing:
        raise FileNotFoundError(f"AACT zip is missing: {sorted(missing)}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m ctrisk.ingest.aact <url-or-zip-path>")
    cfg = load_config()
    for p in fetch(sys.argv[1], Path(cfg.local_root) / "raw" / "aact"):
        print(f"extracted {p}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aact_ingest.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/ctrisk/ingest/aact.py tests/test_aact_ingest.py
git commit -m "Add AACT ingest"
```

---

### Task 4: Fixtures and the trial table

**Files:** Create `tests/fixtures/aact/{studies,designs,sponsors,interventions}.txt`, `tests/conftest.py`, `tests/test_clean_trials.py`, `src/ctrisk/spark/clean_trials.py`

Fixture trials, one per rule. Expected output: **NCT001, NCT002, NCT003, NCT009**.

| Trial | Case | Kept? |
|---|---|---|
| NCT001 | Completed Phase 2 drug, 2012 | yes, label 0 |
| NCT002 | Terminated Phase 3 biologic, 2015; also has a collaborator sponsor | yes, label 1 |
| NCT003 | Recruiting Phase 1, 2024 | yes, label null (scored later) |
| NCT004 | Observational | no |
| NCT005 | Phase 4 | no |
| NCT006 | Terminated but started 2005 | no |
| NCT007 | Withdrawn | no |
| NCT008 | Device-only | no |
| NCT009 | Legacy casing (`Terminated`, `Phase 1/Phase 2`, `Drug`, `Industry`); no design row | yes, label 1 |
| NCT010 | Completed but started 2021 | no |

- [ ] **Step 1: Write fixtures**

`tests/fixtures/aact/studies.txt`
```
nct_id|study_type|overall_status|phase|start_date|number_of_arms
NCT001|INTERVENTIONAL|COMPLETED|PHASE2|2012-03-01|2
NCT002|INTERVENTIONAL|TERMINATED|PHASE3|2015-06-15|2
NCT003|INTERVENTIONAL|RECRUITING|PHASE1|2024-01-10|1
NCT004|OBSERVATIONAL|COMPLETED|NA|2012-01-01|
NCT005|INTERVENTIONAL|COMPLETED|PHASE4|2012-01-01|2
NCT006|INTERVENTIONAL|TERMINATED|PHASE2|2005-01-01|2
NCT007|INTERVENTIONAL|WITHDRAWN|PHASE2|2012-01-01|2
NCT008|INTERVENTIONAL|COMPLETED|PHASE1/PHASE2|2019-11-30|1
NCT009|Interventional|Terminated|Phase 1/Phase 2|2018-05-01|3
NCT010|INTERVENTIONAL|COMPLETED|PHASE2|2021-02-01|2
```

`tests/fixtures/aact/interventions.txt`
```
id|nct_id|intervention_type|name
1|NCT001|DRUG|Pembrolizumab
2|NCT002|BIOLOGICAL|Adalimumab
3|NCT002|OTHER|Placebo
4|NCT003|DRUG|Metformin 500 mg
5|NCT004|DRUG|Aspirin
6|NCT005|DRUG|Aspirin
7|NCT006|DRUG|Aspirin
8|NCT007|DRUG|Aspirin
9|NCT008|DEVICE|Coronary stent
10|NCT009|Drug|Ibuprofen
11|NCT010|DRUG|Aspirin
```

`tests/fixtures/aact/sponsors.txt`
```
id|nct_id|agency_class|lead_or_collaborator|name
1|NCT001|INDUSTRY|lead|Merck
2|NCT002|OTHER|lead|State University
3|NCT002|INDUSTRY|collaborator|AbbVie
4|NCT003|NIH|lead|National Cancer Institute
5|NCT009|Industry|lead|Pfizer
```

`tests/fixtures/aact/designs.txt`
```
id|nct_id|allocation|intervention_model|primary_purpose|masking
1|NCT001|RANDOMIZED|PARALLEL|TREATMENT|DOUBLE
2|NCT002|RANDOMIZED|PARALLEL|TREATMENT|QUADRUPLE
3|NCT003|NA|SINGLE_GROUP|TREATMENT|NONE
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder.master("local[1]")
         .config("spark.sql.shuffle.partitions", "1")
         .config("spark.sql.session.timeZone", "UTC")
         .getOrCreate())
    yield s
    s.stop()


@pytest.fixture(scope="session")
def aact(spark):
    from ctrisk.spark.clean_trials import read_table
    return {t: read_table(spark, str(FIXTURES / "aact"), t)
            for t in ("studies", "designs", "sponsors", "interventions")}
```

- [ ] **Step 3: Write the failing tests**

`tests/test_clean_trials.py`
```python
import pytest

from ctrisk.spark.clean_trials import build_trials


@pytest.fixture(scope="module")
def trials(aact):
    rows = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"]).collect()
    return {r.nct_id: r for r in rows}


def test_keeps_only_eligible_drug_trials(trials):
    assert set(trials) == {"NCT001", "NCT002", "NCT003", "NCT009"}


def test_label_is_terminated_vs_completed_and_null_for_active(trials):
    assert {k: r.label for k, r in trials.items()} == {
        "NCT001": 0, "NCT002": 1, "NCT003": None, "NCT009": 1,
    }


def test_normalizes_legacy_casing(trials):
    assert trials["NCT009"].phase == "PHASE1/PHASE2"
    assert trials["NCT009"].status == "TERMINATED"
    assert trials["NCT009"].sponsor_class == "INDUSTRY"


def test_uses_lead_sponsor_only(trials):
    assert trials["NCT002"].sponsor_class == "OTHER"
    assert trials["NCT002"].sponsor_name == "State University"


def test_maps_nih_to_government(trials):
    assert trials["NCT003"].sponsor_class == "GOVERNMENT"


def test_missing_design_is_null_not_dropped(trials):
    assert trials["NCT009"].masking is None
    assert trials["NCT001"].masking == "DOUBLE"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_clean_trials.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ctrisk.spark.clean_trials'`

- [ ] **Step 5: Implement `src/ctrisk/spark/clean_trials.py`**

```python
"""AACT -> one row per eligible drug trial, with its label.

Population: interventional drug/biologic trials, Phase 1-3.
  Finished (completed/terminated) and started 2008-2020 -> label 0/1, used for training.
  Still active                                         -> label null, scored later.
"""
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

PHASES = ["PHASE1", "PHASE1/PHASE2", "PHASE2", "PHASE2/PHASE3", "PHASE3"]
ACTIVE = ["NOT_YET_RECRUITING", "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]
DRUG_TYPES = ["DRUG", "BIOLOGICAL"]
GOVERNMENT = ["NIH", "FED", "OTHER_GOV"]
DESIGN_COLS = ["allocation", "intervention_model", "primary_purpose", "masking"]
TRAIN_START, TRAIN_END = "2008-01-01", "2020-12-31"


def read_table(spark: SparkSession, aact_dir: str, name: str) -> DataFrame:
    return spark.read.csv(f"{aact_dir}/{name}.txt", sep="|", header=True,
                          multiLine=True, quote='"', escape='"')


def norm(col: Column) -> Column:
    """'Active, not recruiting' -> 'ACTIVE_NOT_RECRUITING'; 'Phase 1/Phase 2' -> 'PHASE1/PHASE2'."""
    c = F.regexp_replace(F.upper(F.trim(col)), r"PHASE\s+", "PHASE")
    return F.regexp_replace(c, r"[^A-Z0-9/]+", "_")


def build_trials(studies: DataFrame, designs: DataFrame, sponsors: DataFrame,
                 interventions: DataFrame) -> DataFrame:
    drug_trials = (interventions
                   .where(norm(F.col("intervention_type")).isin(DRUG_TYPES))
                   .select("nct_id").distinct())

    agency = norm(F.col("agency_class"))
    lead_sponsor = (sponsors
                    .where(F.lower("lead_or_collaborator") == "lead")
                    .select("nct_id",
                            F.col("name").alias("sponsor_name"),
                            F.when(agency == "INDUSTRY", "INDUSTRY")
                             .when(agency.isin(GOVERNMENT), "GOVERNMENT")
                             .otherwise("OTHER").alias("sponsor_class"))
                    .dropDuplicates(["nct_id"]))

    design = designs.select("nct_id", *[norm(F.col(c)).alias(c) for c in DESIGN_COLS])

    s = studies.select(
        "nct_id",
        norm(F.col("study_type")).alias("study_type"),
        norm(F.col("overall_status")).alias("status"),
        norm(F.col("phase")).alias("phase"),
        F.to_date("start_date").alias("start_date"),
        F.col("number_of_arms").cast("int").alias("number_of_arms"),
    )
    finished = (F.col("status").isin("COMPLETED", "TERMINATED")
                & F.col("start_date").between(TRAIN_START, TRAIN_END))
    eligible = (F.col("study_type") == "INTERVENTIONAL") & F.col("phase").isin(PHASES) \
        & (finished | F.col("status").isin(ACTIVE))

    return (s.where(eligible)
            .join(drug_trials, "nct_id")
            .withColumn("label", F.when(F.col("status") == "TERMINATED", 1)
                                  .when(F.col("status") == "COMPLETED", 0).cast("int"))
            .join(design, "nct_id", "left")
            .join(lead_sponsor, "nct_id", "left")
            .drop("study_type"))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_clean_trials.py -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures tests/conftest.py tests/test_clean_trials.py src/ctrisk/spark/clean_trials.py
git commit -m "Build trial table from AACT"
```

---

### Task 5: Drug interventions and the data gate

**Files:** Modify `src/ctrisk/spark/clean_trials.py`, `tests/test_clean_trials.py`

`drug_interventions` is M2's input for drug matching. The gate protects every later stage.

- [ ] **Step 1: Add failing tests to `tests/test_clean_trials.py`**

```python
from ctrisk.gates import DataGateError
from ctrisk.spark.clean_trials import build_drug_interventions, check_trials


def test_drug_interventions_only_for_eligible_trials(aact):
    trials = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])
    rows = build_drug_interventions(aact["interventions"], trials).collect()
    assert sorted((r.nct_id, r.name) for r in rows) == [
        ("NCT001", "Pembrolizumab"),
        ("NCT002", "Adalimumab"),
        ("NCT003", "Metformin 500 mg"),
        ("NCT009", "Ibuprofen"),
    ]


def test_gate_fails_below_minimum_trials(aact):
    trials = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])
    with pytest.raises(DataGateError, match="labeled trials"):
        check_trials(trials, min_trials=100)


def test_gate_fails_on_implausible_termination_rate(aact):
    # Fixture: 2 of 3 labeled trials terminated (67%), outside 5-25%
    trials = build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])
    with pytest.raises(DataGateError, match="termination rate"):
        check_trials(trials, min_trials=1)
```

Move the `from ctrisk.gates ...` and `from ctrisk.spark.clean_trials ...` imports to the top of the file with the existing imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_clean_trials.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_drug_interventions'`

- [ ] **Step 3: Add to `src/ctrisk/spark/clean_trials.py`**

Add `from ctrisk.gates import DataGateError` to the imports, then append:

```python
def build_drug_interventions(interventions: DataFrame, trials: DataFrame) -> DataFrame:
    return (interventions
            .where(norm(F.col("intervention_type")).isin(DRUG_TYPES))
            .join(trials.select("nct_id"), "nct_id", "left_semi")
            .select("nct_id", F.col("id").alias("intervention_id"), "name"))


def check_trials(trials: DataFrame, min_trials: int) -> dict:
    labeled = trials.where(F.col("label").isNotNull())
    n = labeled.count()
    if n < min_trials:
        raise DataGateError(f"{n} labeled trials, expected at least {min_trials}")
    rate = labeled.agg(F.avg("label")).first()[0]
    if not 0.05 <= rate <= 0.25:
        raise DataGateError(f"termination rate {rate:.1%} is outside 5-25%; check the label logic")
    return {"labeled": n, "termination_rate": round(rate, 3),
            "active": trials.where(F.col("label").isNull()).count()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/ctrisk/spark/clean_trials.py tests/test_clean_trials.py
git commit -m "Add drug interventions table and trial data gate"
```

---

### Task 6: Entry point and CI

**Files:** Modify `src/ctrisk/spark/clean_trials.py`; create `.github/workflows/ci.yml`

- [ ] **Step 1: Append the entry point to `src/ctrisk/spark/clean_trials.py`**

Add `from ctrisk.config import load_config` and `from ctrisk.spark.session import get_spark` to the imports, then append:

```python
if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("clean_trials", cfg.mode)
    t = {name: read_table(spark, cfg.path("raw", "aact"), name)
         for name in ("studies", "designs", "sponsors", "interventions")}

    trials = build_trials(t["studies"], t["designs"], t["sponsors"], t["interventions"]).cache()
    summary = check_trials(trials, cfg.min_trials)

    trials.write.mode("overwrite").parquet(cfg.path("parquet", "trials"))
    build_drug_interventions(t["interventions"], trials) \
        .write.mode("overwrite").parquet(cfg.path("parquet", "drug_interventions"))
    print(summary)
```

- [ ] **Step 2: Write `.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"
      - uses: astral-sh/setup-uv@v6
      - run: uv sync
      - run: uv run pytest -q
```

- [ ] **Step 3: Run the full suite and lint**

```bash
uv run pytest -q
uv run ruff check src tests
```

Expected: `11 passed`; ruff reports no errors.

- [ ] **Step 4: Commit**

```bash
git add src/ctrisk/spark/clean_trials.py .github/workflows/ci.yml
git commit -m "Add trials entry point and CI"
```

---

### Task 7: Run on the real AACT snapshot

**Files:** Possibly modify `tests/fixtures/aact/*.txt` and constants in `src/ctrisk/spark/clean_trials.py`

- [ ] **Step 1: Download the snapshot**

Frank downloads the latest "Flat Text Files" zip (~2.3 GB) from https://aact.ctti-clinicaltrials.org/snapshots in a browser, then:

```bash
cp .env.example .env
make ingest-aact AACT=~/Downloads/<YYYYMMDD>_export_ctgov.zip
```

Expected: four `extracted data/raw/aact/*.txt` lines.

- [ ] **Step 2: Confirm the columns and values the code relies on**

```bash
for t in studies designs sponsors interventions; do echo "== $t"; head -1 data/raw/aact/$t.txt | tr '|' '\n' | grep -nxE 'nct_id|id|study_type|overall_status|phase|start_date|number_of_arms|allocation|intervention_model|primary_purpose|masking|agency_class|lead_or_collaborator|name|intervention_type'; done
cut -d'|' -f1-40 data/raw/aact/studies.txt | head -1
```

Then list the distinct raw values the filters depend on:

```bash
uv run python - <<'PY'
from pyspark.sql import SparkSession
from ctrisk.spark.clean_trials import read_table
spark = SparkSession.builder.master("local[*]").getOrCreate()
for table, col in [("studies", "phase"), ("studies", "overall_status"), ("studies", "study_type"),
                   ("interventions", "intervention_type"), ("sponsors", "agency_class")]:
    print(table, col, sorted(r[0] or "" for r in read_table(spark, "data/raw/aact", table).select(col).distinct().collect()))
PY
```

Check each against `PHASES`, `ACTIVE`, `DRUG_TYPES`, `GOVERNMENT` (after `norm`). If any name or value differs, update the fixture and constant to match, re-run `make test`, and commit with message `Align AACT fields with <date> snapshot`.

- [ ] **Step 3: Build the trial table**

```bash
make trials
```

Expected: a summary like `{'labeled': <n>, 'termination_rate': <r>, 'active': <m>}` and Parquet under `data/parquet/`. If the gate fails, stop and investigate — do not lower the threshold to pass.

- [ ] **Step 4: Record the numbers**

Add under `## Status` in `README.md`: labeled trial count, termination rate, active trial count, and snapshot date. Commit: `Record M1 trial counts`.

---

## After M1

Each milestone gets its own plan once the previous one's real numbers are in:

| Next | Depends on M1 for |
|---|---|
| M2 FAERS ingest + flatten + drug matching | `drug_interventions` names and volume |
| M3 GCS upload + Snowflake stage load + SQL layers + features | `trials` schema; eligibility/country tables added to `TABLES` |
| M4 Train + score + versioning | Labeled count and termination rate |
| M5 Spark on Dataproc | Working local pipeline |
