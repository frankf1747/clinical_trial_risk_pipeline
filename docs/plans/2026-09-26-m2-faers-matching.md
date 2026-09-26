# M2: FAERS Ingest, Flatten, and Drug Matching — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do not commit** — Frank commits this repo himself. End each task by listing changed files.

**Goal:** Download a FAERS sample, flatten it to one row per (report, substance), and map each trial to the FAERS substances it tests — with a reported match rate.

**Architecture:** `ingest/faers.py` picks quarters from openFDA's file list and downloads zips (shared `ingest/download.py`). `spark/flatten_faers.py` reads zips with `binaryFiles` and stream-parses each with `ijson` inside Spark tasks (each file is one large JSON document, not JSON Lines), keeping only the latest version of each report. `drugnames.py` holds pure-Python name normalization and matching, used by both sides. `spark/match_drugs.py` gathers each trial's drug names from three AACT sources and matches them to the FAERS vocabulary.

**Tech Stack:** Python 3.11, PySpark 3.5, ijson, requests, pytest.

**Spec:** `docs/specs/2026-09-26-design.md` · **Sample (Option A):** Q1 of each year 2004–2020 = 279 files, 14.9 GB zipped, ~83 GB JSON, 3.25M reports.

---

## File map

| File | Responsibility |
|---|---|
| `src/ctrisk/ingest/download.py` | Download once: `.part` → zip check → rename (moved out of `aact.py`) |
| `src/ctrisk/ingest/aact.py` | Use shared download; two more tables |
| `src/ctrisk/ingest/faers.py` | Select FAERS quarters from `download.json`; parallel download |
| `src/ctrisk/drugnames.py` | `normalize(name)`, `match_substances(name, vocab)` |
| `src/ctrisk/spark/flatten_faers.py` | Zips → `faers_drug_events` Parquet |
| `src/ctrisk/spark/match_drugs.py` | Trial names + FAERS vocab → `trial_drug_map` Parquet + match-rate gate |
| `src/ctrisk/checks/faers_api.py` | Spot-check our counts against the openFDA API |
| `src/ctrisk/config.py`, `src/ctrisk/spark/session.py` | FAERS settings; local driver memory |
| `tests/test_download.py`, `tests/test_drugnames.py`, `tests/test_faers_ingest.py`, `tests/test_flatten_faers.py`, `tests/test_match_drugs.py` | Tests |
| `tests/fixtures/aact/{intervention_other_names,browse_interventions}.txt` | New fixtures |
| `Makefile`, `.env.example`, `README.md`, `pyproject.toml` | Wiring |

---

### Task 1: Shared download helper

**Files:** Create `src/ctrisk/ingest/download.py`, `tests/test_download.py`; modify `src/ctrisk/ingest/aact.py`

- [ ] **Step 1: Write failing tests** — `tests/test_download.py`

```python
import io
import zipfile

import pytest

from ctrisk.ingest import download as dl


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.body


def zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x")
    return buf.getvalue()


def test_saves_zip_and_leaves_no_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResponse(zip_bytes()))
    out = dl.download("https://example.org/a.zip", tmp_path / "q1" / "a.zip")
    assert zipfile.is_zipfile(out)
    assert not list(tmp_path.rglob("*.part"))


def test_rejects_non_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResponse(b"<html>blocked</html>"))
    with pytest.raises(ValueError, match="not return a zip"):
        dl.download("https://example.org/a.zip", tmp_path / "a.zip")
    assert not list(tmp_path.iterdir())


def test_skips_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "a.zip"
    target.write_bytes(b"already here")
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: pytest.fail("should not download"))
    assert dl.download("https://example.org/a.zip", target) == target
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL — `ImportError: cannot import name 'download' from 'ctrisk.ingest'`

- [ ] **Step 3: Implement** — `src/ctrisk/ingest/download.py`

```python
"""Download a zip once: stream to a .part file, confirm it is a zip, then rename."""
import zipfile
from pathlib import Path

import requests


def download(url: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(partial, "wb") as f:
            f.writelines(r.iter_content(1 << 20))
    if not zipfile.is_zipfile(partial):
        partial.unlink()
        raise ValueError(f"{url} did not return a zip; download it in a browser and pass the local path")
    partial.rename(target)
    return target
```

- [ ] **Step 4: Use it in `src/ctrisk/ingest/aact.py`**

Delete `_download` and the `import requests` line, add `from ctrisk.ingest.download import download`, and change the `zip_path` line in `fetch` to:

```python
    zip_path = download(source, dest.parent / "aact.zip") if is_url else Path(source)
```

Also extend `TABLES` (M2 needs these two):

```python
TABLES = ("studies", "designs", "sponsors", "interventions",
          "intervention_other_names", "browse_interventions")
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass (15), ruff clean.

---

### Task 2: Drug name normalization and matching

**Files:** Create `src/ctrisk/drugnames.py`, `tests/test_drugnames.py`

Pure Python so both the FAERS parser and the Spark matcher use the same rules.

- [ ] **Step 1: Write failing tests** — `tests/test_drugnames.py`

```python
import pytest

from ctrisk.drugnames import match_substances, normalize


@pytest.mark.parametrize("raw, expected", [
    ("Metformin 500 mg", "metformin"),
    ("doxorubicin hydrochloride", "doxorubicin"),
    ("ONDANSETRON HYDROCHLORIDE", "ondansetron"),
    ("5-fluorouracil", "fluorouracil"),
    ("Cisplatin 75 mg/m2 IV infusion", "cisplatin"),
    ("Insulin Glargine", "insulin glargine"),
    ("Placebo", ""),
    ("Sodium chloride 0.9%", ""),
    ("3 lines of therapy", "lines of therapy"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


VOCAB = frozenset({"pembrolizumab", "insulin", "insulin glargine", "iron", "tea"})


def test_matches_longest_phrase_first():
    assert match_substances("insulin glargine", VOCAB) == ["insulin glargine"]


def test_matches_several_substances_in_one_name():
    assert match_substances("pembrolizumab plus insulin glargine", VOCAB) == [
        "insulin glargine", "pembrolizumab"]


def test_ignores_short_single_words():
    assert match_substances("green tea extract", VOCAB) == []
    assert match_substances("iron", VOCAB) == ["iron"]


def test_no_match():
    assert match_substances("", VOCAB) == []
    assert match_substances("exercise program", VOCAB) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_drugnames.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ctrisk.drugnames'`

- [ ] **Step 3: Implement** — `src/ctrisk/drugnames.py`

```python
"""Normalize drug names so trial interventions and FAERS substances compare equal.

'Cisplatin 75 mg/m2 IV infusion' -> 'cisplatin';  'ONDANSETRON HYDROCHLORIDE' -> 'ondansetron'
"""
import re

_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|kg|ml|l|iu|units?|%)(?:/\S+)?(?![a-z])")
_NOISE = frozenset("""
    hydrochloride hcl sodium potassium calcium magnesium sulfate sulphate acetate maleate mesylate
    tartrate citrate phosphate succinate besylate fumarate bromide chloride hydrobromide disodium
    monohydrate dihydrate anhydrous
    tablet tablets capsule capsules injection injectable infusion oral iv intravenous subcutaneous
    intramuscular topical cream gel ointment solution suspension patch spray inhaled inhalation
    extended release er xr sr dose dosing
    placebo saline vehicle sham
""".split())
MIN_SINGLE_WORD = 4  # a one-word match shorter than this ("tea") is too likely to be noise


def normalize(name: str) -> str:
    text = _DOSE.sub(" ", name.lower())
    tokens = re.sub(r"[^a-z0-9]+", " ", text).split()
    return " ".join(t for t in tokens if t not in _NOISE and not t.isdigit())


def match_substances(name: str, vocab: frozenset[str], max_words: int = 4) -> list[str]:
    """Substances from `vocab` in a normalized name, longest phrase first, no overlaps."""
    words = name.split()
    found, i = set(), 0
    while i < len(words):
        for n in range(min(max_words, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            if phrase in vocab and (n > 1 or len(phrase) >= MIN_SINGLE_WORD):
                found.add(phrase)
                i += n
                break
        else:
            i += 1
    return sorted(found)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_drugnames.py -v && uv run ruff check src tests`
Expected: all pass; ruff clean.

---

### Task 3: FAERS settings and ingest

**Files:** Modify `pyproject.toml`, `src/ctrisk/config.py`, `.env.example`; create `src/ctrisk/ingest/faers.py`, `tests/test_faers_ingest.py`

- [ ] **Step 1: Add ijson**

Run: `uv add "ijson>=3.3"`

- [ ] **Step 2: Replace `src/ctrisk/config.py`**

```python
"""Settings from .env. Everything that differs between local and cloud lives here."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    mode: str                     # "local" or "cloud"
    local_root: str               # where ingest writes downloads
    data_root: str                # where Spark reads and writes
    min_trials: int
    faers_years: range
    faers_quarters: frozenset[int]
    min_match_rate: float

    def path(self, *parts: str) -> str:
        return "/".join([self.data_root, *parts])


def load_config() -> Config:
    load_dotenv()
    mode = os.getenv("MODE", "local")
    local_root = os.getenv("DATA_DIR", "data")
    if mode == "local":
        data_root = local_root
    elif mode == "cloud":
        bucket = os.getenv("GCP_BUCKET")
        if not bucket:
            raise ValueError("GCP_BUCKET is required when MODE=cloud")
        data_root = f"gs://{bucket}"
    else:
        raise ValueError(f"MODE must be 'local' or 'cloud', got {mode!r}")
    first, last = os.getenv("FAERS_YEARS", "2004-2020").split("-")
    return Config(
        mode=mode,
        local_root=local_root,
        data_root=data_root,
        min_trials=int(os.getenv("MIN_TRIALS", "40000")),
        faers_years=range(int(first), int(last) + 1),
        faers_quarters=frozenset(int(q) for q in os.getenv("FAERS_QUARTERS", "1").split(",")),
        min_match_rate=float(os.getenv("MIN_MATCH_RATE", "0")),
    )
```

- [ ] **Step 3: Update `.env.example`** — insert after `MIN_TRIALS=40000`:

```bash
MIN_MATCH_RATE=0          # set in M2 Task 7 from the first real run

# FAERS sample: Q1 of each year 2004-2020. Full history: FAERS_YEARS=2004-2026, FAERS_QUARTERS=1,2,3,4
FAERS_YEARS=2004-2020
FAERS_QUARTERS=1
```

- [ ] **Step 4: Write failing tests** — `tests/test_faers_ingest.py`

```python
from pathlib import Path

from ctrisk.ingest.faers import local_path, select_partitions

BASE = "https://download.open.fda.gov/drug/event"
PARTITIONS = [
    {"file": f"{BASE}/2004q1/drug-event-0001-of-0005.json.zip"},
    {"file": f"{BASE}/2004q2/drug-event-0001-of-0005.json.zip"},
    {"file": f"{BASE}/2020q1/drug-event-0003-of-0032.json.zip"},
    {"file": f"{BASE}/2021q1/drug-event-0001-of-0030.json.zip"},
    {"file": f"{BASE}/all_other/drug-event-0001-of-0004.json.zip"},
]


def test_selects_requested_years_and_quarters():
    urls = select_partitions(PARTITIONS, range(2004, 2021), frozenset({1}))
    assert urls == [f"{BASE}/2004q1/drug-event-0001-of-0005.json.zip",
                    f"{BASE}/2020q1/drug-event-0003-of-0032.json.zip"]


def test_local_path_keeps_quarter_folder():
    url = f"{BASE}/2020q1/drug-event-0003-of-0032.json.zip"
    assert local_path(url, Path("data/raw/faers")) == Path("data/raw/faers/2020q1/drug-event-0003-of-0032.json.zip")
```

- [ ] **Step 5: Run to verify failure**

Run: `uv run pytest tests/test_faers_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ctrisk.ingest.faers'`

- [ ] **Step 6: Implement** — `src/ctrisk/ingest/faers.py`

```python
"""Download the selected FAERS drug/event quarters listed in openFDA's download.json."""
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from ctrisk.config import load_config
from ctrisk.ingest.download import download

INDEX_URL = "https://api.fda.gov/download.json"


def select_partitions(partitions: list[dict], years: range, quarters: frozenset[int]) -> list[str]:
    urls = []
    for p in partitions:
        m = re.search(r"/(\d{4})q(\d)/", p["file"])
        if m and int(m[1]) in years and int(m[2]) in quarters:
            urls.append(p["file"])
    return urls


def local_path(url: str, root: Path) -> Path:
    quarter, name = url.split("/")[-2:]
    return root / quarter / name


if __name__ == "__main__":
    cfg = load_config()
    partitions = requests.get(INDEX_URL, timeout=60).json()["results"]["drug"]["event"]["partitions"]
    urls = select_partitions(partitions, cfg.faers_years, cfg.faers_quarters)
    root = Path(cfg.local_root) / "raw" / "faers"
    print(f"{len(urls)} FAERS files -> {root}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(lambda u: download(u, local_path(u, root)), urls), 1):
            if i % 25 == 0 or i == len(urls):
                print(f"{i}/{len(urls)}")
```

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass; ruff clean.

---

### Task 4: Flatten FAERS

**Files:** Modify `src/ctrisk/spark/session.py`; create `src/ctrisk/spark/flatten_faers.py`, `tests/test_flatten_faers.py`

Output `faers_drug_events`: `safetyreportid, receivedate (date), serious, death, substance, suspect, harmonized`. `suspect` = the reporter named this drug as a suspected cause. `harmonized` = the name came from openFDA's `substance_name` (used to build the match vocabulary).

- [ ] **Step 1: Give local Spark more memory** — `src/ctrisk/spark/session.py`

```python
from pyspark.sql import SparkSession


def get_spark(app: str, mode: str = "local") -> SparkSession:
    builder = SparkSession.builder.appName(app).config("spark.sql.session.timeZone", "UTC")
    if mode == "local":
        # on Dataproc the cluster sets the master and memory
        builder = builder.master("local[*]").config("spark.driver.memory", "6g")
    return builder.getOrCreate()
```

- [ ] **Step 2: Write failing tests** — `tests/test_flatten_faers.py`

```python
import json
import zipfile

from ctrisk.spark.flatten_faers import build_drug_events, report_rows


def report(rid, version="1", serious="1", death=None, drugs=()):
    r = {"safetyreportid": rid, "safetyreportversion": version, "receivedate": "20160107",
         "serious": serious, "patient": {"drug": list(drugs)}}
    if death:
        r["seriousnessdeath"] = death
    return r


ONDANSETRON = {"drugcharacterization": "1", "medicinalproduct": "ZOFRAN",
               "openfda": {"substance_name": ["ONDANSETRON HYDROCHLORIDE", "ONDANSETRON"]}}
NO_OPENFDA = {"drugcharacterization": "2", "medicinalproduct": "Some Brand",
              "activesubstance": {"activesubstancename": "METFORMIN HYDROCHLORIDE"}}


def test_salt_variants_collapse_to_one_substance():
    rows = list(report_rows(report("1", drugs=[ONDANSETRON])))
    assert rows == [("1", 1, "20160107", True, False, True, True, "ondansetron")]


def test_falls_back_to_active_substance_and_flags_it():
    rows = list(report_rows(report("2", serious="2", death="1", drugs=[NO_OPENFDA])))
    assert rows == [("2", 1, "20160107", False, True, False, False, "metformin")]


def write_zip(path, reports):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("drug-event.json", json.dumps({"meta": {}, "results": reports}))


def test_keeps_latest_report_version_only(spark, tmp_path):
    (tmp_path / "2016q1").mkdir()
    write_zip(tmp_path / "2016q1" / "a.zip", [report("9", version="1", serious="2", drugs=[ONDANSETRON])])
    write_zip(tmp_path / "2016q1" / "b.zip", [report("9", version="2", serious="1", drugs=[ONDANSETRON]),
                                              report("10", drugs=[NO_OPENFDA])])

    rows = {r.safetyreportid: r for r in build_drug_events(spark, str(tmp_path / "*" / "*.zip")).collect()}

    assert set(rows) == {"9", "10"}
    assert rows["9"].serious is True               # version 2 wins
    assert str(rows["9"].receivedate) == "2016-01-07"
    assert rows["10"].substance == "metformin"
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_flatten_faers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ctrisk.spark.flatten_faers'`

- [ ] **Step 4: Implement** — `src/ctrisk/spark/flatten_faers.py`

```python
"""FAERS bulk zips -> one row per (report, substance), latest report version only.

Each openFDA file is a single JSON document ({"meta":..., "results":[...]}), so each Spark task
streams one zip with ijson instead of loading it whole.
"""
import io
import zipfile
from collections.abc import Iterator

import ijson
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ctrisk.config import load_config
from ctrisk.drugnames import normalize
from ctrisk.spark.session import get_spark

SCHEMA = ("safetyreportid string, version int, receivedate string, serious boolean, "
          "death boolean, suspect boolean, harmonized boolean, substance string")


def report_rows(report: dict) -> Iterator[tuple]:
    base = (report["safetyreportid"], int(report.get("safetyreportversion") or 1),
            report.get("receivedate"), report.get("serious") == "1",
            report.get("seriousnessdeath") == "1")
    for drug in report.get("patient", {}).get("drug", []):
        harmonized = drug.get("openfda", {}).get("substance_name")
        fallback = (drug.get("activesubstance") or {}).get("activesubstancename") \
            or drug.get("medicinalproduct") or ""
        for substance in sorted({normalize(n) for n in harmonized or [fallback]} - {""}):
            yield (*base, drug.get("drugcharacterization") == "1", bool(harmonized), substance)


def parse_zip(content: bytes) -> Iterator[tuple]:
    with zipfile.ZipFile(io.BytesIO(content)) as z, z.open(z.namelist()[0]) as f:
        for report in ijson.items(f, "results.item"):
            yield from report_rows(report)


def build_drug_events(spark: SparkSession, zip_glob: str) -> DataFrame:
    rows = spark.sparkContext.binaryFiles(zip_glob).flatMap(lambda kv: parse_zip(kv[1]))
    latest = F.max("version").over(Window.partitionBy("safetyreportid"))
    return (spark.createDataFrame(rows, SCHEMA)
            .withColumn("latest", latest)  # window functions can't go directly in where()
            .where(F.col("version") == F.col("latest"))
            .groupBy("safetyreportid", "receivedate", "serious", "death", "substance")
            .agg(F.max("suspect").alias("suspect"), F.max("harmonized").alias("harmonized"))
            .withColumn("receivedate", F.to_date("receivedate", "yyyyMMdd")))


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("flatten_faers", cfg.mode)
    out = cfg.path("parquet", "faers_drug_events")
    build_drug_events(spark, cfg.path("raw", "faers", "*", "*.zip")).write.mode("overwrite").parquet(out)
    events = spark.read.parquet(out)
    print({"rows": events.count(),
           "reports": events.select("safetyreportid").distinct().count(),
           "substances": events.select("substance").distinct().count()})
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass; ruff clean.

---

### Task 5: Match trial drugs to FAERS substances

**Files:** Create `tests/fixtures/aact/intervention_other_names.txt`, `tests/fixtures/aact/browse_interventions.txt`, `src/ctrisk/spark/match_drugs.py`, `tests/test_match_drugs.py`; modify `tests/conftest.py`

Fixture design (drug interventions from M1 fixtures: NCT001 Pembrolizumab id 1, NCT002 Adalimumab id 2, NCT003 Metformin id 4, NCT009 Ibuprofen id 10; intervention id 3 is NCT002's Placebo, type OTHER):

| Row | Purpose |
|---|---|
| other name `MK-3475` for intervention 1 | brand/code names are included |
| other name `Sugar pill` for intervention 3 | names of non-drug interventions are excluded |
| MeSH `Ibuprofen` (mesh-list) for NCT009 | MeSH terms are included |
| MeSH ancestor for NCT009 | ancestors are excluded |
| MeSH `Aspirin` for NCT004 | ineligible trials are excluded |

- [ ] **Step 1: Write fixtures**

`tests/fixtures/aact/intervention_other_names.txt`
```
id|nct_id|intervention_id|name
1|NCT001|1|MK-3475
2|NCT002|3|Sugar pill
```

`tests/fixtures/aact/browse_interventions.txt`
```
id|nct_id|mesh_term|downcase_mesh_term|mesh_type
1|NCT009|Ibuprofen|ibuprofen|mesh-list
2|NCT009|Anti-Inflammatory Agents, Non-Steroidal|anti-inflammatory agents, non-steroidal|mesh-ancestor
3|NCT004|Aspirin|aspirin|mesh-list
```

- [ ] **Step 2: Load them in `tests/conftest.py`** — change the `aact` fixture's table tuple to:

```python
            for t in ("studies", "designs", "sponsors", "interventions",
                      "intervention_other_names", "browse_interventions")}
```

- [ ] **Step 3: Write failing tests** — `tests/test_match_drugs.py`

```python
import pytest

from ctrisk.gates import DataGateError
from ctrisk.spark.clean_trials import build_drug_interventions, build_trials
from ctrisk.spark.match_drugs import build_trial_drug_map, check_match_rate, trial_drug_names

VOCAB = frozenset({"pembrolizumab", "metformin", "ibuprofen"})  # adalimumab deliberately absent


@pytest.fixture(scope="module")
def trials(aact):
    return build_trials(aact["studies"], aact["designs"], aact["sponsors"], aact["interventions"])


@pytest.fixture(scope="module")
def names(aact, trials):
    drugs = build_drug_interventions(aact["interventions"], trials)
    return trial_drug_names(drugs, aact["intervention_other_names"], aact["browse_interventions"])


def test_names_come_from_three_sources(names):
    got = {(r.nct_id, r.name) for r in names.collect()}
    assert got == {
        ("NCT001", "Pembrolizumab"), ("NCT001", "MK-3475"),
        ("NCT002", "Adalimumab"),
        ("NCT003", "Metformin 500 mg"),
        ("NCT009", "Ibuprofen"),
    }


def test_maps_trials_to_substances(names):
    got = {(r.nct_id, r.substance) for r in build_trial_drug_map(names, VOCAB).collect()}
    assert got == {("NCT001", "pembrolizumab"), ("NCT003", "metformin"), ("NCT009", "ibuprofen")}


def test_match_rate_counts_labeled_trials(trials, names):
    # labeled: NCT001, NCT002, NCT009 -> NCT001 and NCT009 matched
    summary = check_match_rate(trials, build_trial_drug_map(names, VOCAB), min_rate=0.5)
    assert summary == {"labeled_trials": 3, "matched": 2, "match_rate": 0.667, "substances": 3}


def test_match_rate_gate(trials, names):
    with pytest.raises(DataGateError, match="match rate"):
        check_match_rate(trials, build_trial_drug_map(names, VOCAB), min_rate=0.9)
```

- [ ] **Step 4: Run to verify failure**

Run: `uv run pytest tests/test_match_drugs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ctrisk.spark.match_drugs'`

- [ ] **Step 5: Implement** — `src/ctrisk/spark/match_drugs.py`

```python
"""Map each trial to the FAERS substances it tests -> trial_drug_map(nct_id, substance)."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ctrisk.config import load_config
from ctrisk.drugnames import match_substances, normalize
from ctrisk.gates import DataGateError
from ctrisk.spark.clean_trials import read_table
from ctrisk.spark.session import get_spark


def trial_drug_names(drug_interventions: DataFrame, other_names: DataFrame,
                     browse_interventions: DataFrame) -> DataFrame:
    """Every name a trial uses for its drugs: intervention names, other names, MeSH terms."""
    trials = drug_interventions.select("nct_id").distinct()
    others = (other_names
              .join(drug_interventions.select("intervention_id"), "intervention_id", "left_semi")
              .select("nct_id", "name"))
    mesh = (browse_interventions
            .where(F.col("mesh_type") == "mesh-list")
            .join(trials, "nct_id", "left_semi")
            .select("nct_id", F.col("mesh_term").alias("name")))
    return drug_interventions.select("nct_id", "name").unionByName(others).unionByName(mesh).distinct()


def faers_vocab(drug_events: DataFrame) -> frozenset[str]:
    rows = drug_events.where("harmonized").select("substance").distinct().collect()
    return frozenset(r.substance for r in rows)


def build_trial_drug_map(names: DataFrame, vocab: frozenset[str]) -> DataFrame:
    shared = names.sparkSession.sparkContext.broadcast(vocab)

    @F.udf(T.ArrayType(T.StringType()))
    def substances(name):
        return match_substances(normalize(name or ""), shared.value)

    return names.select("nct_id", F.explode(substances("name")).alias("substance")).distinct()


def check_match_rate(trials: DataFrame, trial_drug_map: DataFrame, min_rate: float) -> dict:
    labeled = trials.where(F.col("label").isNotNull()).select("nct_id")
    total = labeled.count()
    matched = labeled.join(trial_drug_map, "nct_id", "left_semi").count()
    rate = matched / total if total else 0.0
    if rate < min_rate:
        raise DataGateError(f"drug match rate {rate:.1%} is below the {min_rate:.0%} minimum")
    return {"labeled_trials": total, "matched": matched, "match_rate": round(rate, 3),
            "substances": trial_drug_map.select("substance").distinct().count()}


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("match_drugs", cfg.mode)
    aact = cfg.path("raw", "aact")
    names = trial_drug_names(spark.read.parquet(cfg.path("parquet", "drug_interventions")),
                             read_table(spark, aact, "intervention_other_names"),
                             read_table(spark, aact, "browse_interventions"))
    vocab = faers_vocab(spark.read.parquet(cfg.path("parquet", "faers_drug_events")))
    drug_map = build_trial_drug_map(names, vocab).cache()
    summary = check_match_rate(spark.read.parquet(cfg.path("parquet", "trials")), drug_map,
                               cfg.min_match_rate)
    drug_map.write.mode("overwrite").parquet(cfg.path("parquet", "trial_drug_map"))
    print({"vocab": len(vocab), **summary})
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass; ruff clean.

---

### Task 6: API spot-check and wiring

**Files:** Create `src/ctrisk/checks/__init__.py` (empty), `src/ctrisk/checks/faers_api.py`; modify `Makefile`, `README.md`

The spot-check needs the network, so it is a manual command, not a unit test.

- [ ] **Step 1: Implement** — `src/ctrisk/checks/faers_api.py`

```python
"""Compare our FAERS report counts with the openFDA API for a few substances in one quarter.

openFDA's search matches words inside substance names, so the API can count slightly more
(e.g. salt variants). A ratio near 1.0 means ingest and flatten lost nothing.
"""
import requests
from pyspark.sql import functions as F

from ctrisk.config import load_config
from ctrisk.spark.session import get_spark

API = "https://api.fda.gov/drug/event.json"
SUBSTANCES = ["ondansetron", "metformin", "atorvastatin", "adalimumab"]
START, END = "2016-01-01", "2016-03-31"


def api_count(substance: str) -> int:
    search = (f"receivedate:[{START.replace('-', '')} TO {END.replace('-', '')}]"
              f' AND patient.drug.openfda.substance_name:"{substance}"')
    r = requests.get(API, params={"search": search, "limit": 1}, timeout=60)
    if r.status_code == 404:  # openFDA answers 404 when nothing matches
        return 0
    r.raise_for_status()
    return r.json()["meta"]["results"]["total"]


if __name__ == "__main__":
    cfg = load_config()
    spark = get_spark("check_faers", cfg.mode)
    ours = dict(spark.read.parquet(cfg.path("parquet", "faers_drug_events"))
                .where(F.col("harmonized") & F.col("receivedate").between(START, END)
                       & F.col("substance").isin(SUBSTANCES))
                .groupBy("substance").agg(F.countDistinct("safetyreportid"))
                .collect())
    print(f"{'substance':<14}{'ours':>8}{'api':>8}{'ratio':>8}")
    for s in SUBSTANCES:
        api = api_count(s)
        print(f"{s:<14}{ours.get(s, 0):>8}{api:>8}{ours.get(s, 0) / api if api else 0:>8.2f}")
```

- [ ] **Step 2: Replace `Makefile`**

```make
.PHONY: setup test ingest-aact ingest-faers trials faers match check-faers

setup:
	uv sync

test:
	uv run pytest -q

# Usage: make ingest-aact AACT=<url-or-path-to-aact-zip>
ingest-aact:
	uv run python -m ctrisk.ingest.aact $(AACT)

ingest-faers:
	uv run python -m ctrisk.ingest.faers

trials:
	uv run python -m ctrisk.spark.clean_trials

faers:
	uv run python -m ctrisk.spark.flatten_faers

match:
	uv run python -m ctrisk.spark.match_drugs

check-faers:
	uv run python -m ctrisk.checks.faers_api
```

- [ ] **Step 3: Update the README `## Run` block**

```bash
make setup
make ingest-aact AACT=<aact-flat-files.zip>   # from https://aact.ctti-clinicaltrials.org/snapshots
make ingest-faers                             # FAERS sample set in .env (default: Q1 2004-2020, 14.9 GB)
make trials                                   # -> data/parquet/trials, drug_interventions
make faers                                    # -> data/parquet/faers_drug_events
make match                                    # -> data/parquet/trial_drug_map
make check-faers                              # our counts vs the openFDA API
make test
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all pass; ruff clean.

---

### Task 7: Run on real data

Needs network and ~100 GB free disk (zips 15 GB + Parquet). No code changes unless a check fails.

- [ ] **Step 1: Re-extract AACT with the two new tables, rebuild trials**

```bash
make ingest-aact AACT=data/raw/aact_20260926.zip
make trials
```

Expected: 6 `extracted` lines; same summary as M1 (`labeled 66124`).

- [ ] **Step 2: Download FAERS**

```bash
make ingest-faers
```

Expected: `279 FAERS files`, progress to `279/279`. Rerunning skips finished files.

- [ ] **Step 3: Flatten** (expect roughly 20–60 min locally)

```bash
make faers
```

Expected: `reports` close to 3.25M (slightly lower after dropping old versions); record the printed numbers.

- [ ] **Step 4: Spot-check against the API**

```bash
make check-faers
```

Expected: ratios roughly 0.9–1.0. A ratio far below means reports were dropped: stop and investigate before matching.

- [ ] **Step 5: Match with the gate off, then inspect**

```bash
make match
```

Record `match_rate`. Then look at the 30 most common unmatched trial names to judge whether misses are real non-drugs (e.g. "chemotherapy", "standard of care") or fixable normalization gaps:

```bash
uv run python - <<'PY'
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.master("local[*]").getOrCreate()
d = spark.read.parquet("data/parquet/drug_interventions")
m = spark.read.parquet("data/parquet/trial_drug_map")
(d.join(m, "nct_id", "left_anti").groupBy(F.lower("name").alias("name")).count()
  .orderBy(F.desc("count")).show(30, truncate=False))
PY
```

Report the match rate and the list to Frank before changing any rule. Rule changes go in `drugnames.py` with a test case per fix.

- [ ] **Step 6: Set the gate and record results**

With Frank's agreement, set `MIN_MATCH_RATE` in `.env.example` to the observed rate rounded down to the nearest 5 points, and add to the README `## Status` section: FAERS files/GB, reports, substances, match rate, API-check ratios.

---

## Self-review notes

- `binaryFiles` loads one whole zip per task (FAERS zips are up to ~100 MB); ijson then streams the JSON inside it, so memory stays bounded.
- Dataproc (M5) will need `ijson` on workers (`--properties dataproc:pip.packages=ijson==<ver>`).
- FAERS `receivedate` is what M3 uses for point-in-time counts; nothing in M2 filters by trial dates.
