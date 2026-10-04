"""The Cloud Run FAERS ingest job's pure parts (no network, no GCS)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest_job"))
import main as job

PARTS = [{"file": f"https://download.open.fda.gov/drug/event/{q}/drug-event-000{i}-of-0003.json.zip",
          "size_mb": "10.5"} for q in ("2004q1", "2004q2", "2026q1") for i in (1, 2, 3)]


def test_object_name_keeps_the_quarter_folder_layout_flatten_faers_reads():
    assert job.object_name(PARTS[0]["file"], "raw/faers") == "raw/faers/2004q1/drug-event-0001-of-0003.json.zip"


def test_tasks_split_the_files_without_overlap_or_gaps():
    shares = [job.share(PARTS, task, 4) for task in range(4)]
    names = [p["file"] for s in shares for p in s]
    assert sorted(names) == sorted(p["file"] for p in PARTS) and len(names) == len(set(names))
    assert max(map(len, shares)) - min(map(len, shares)) <= 1


def test_only_files_missing_from_gcs_are_fetched():
    listed = {"raw/faers/2004q1/drug-event-0001-of-0003.json.zip"}
    todo = job.missing(PARTS, "raw/faers", listed)
    assert PARTS[0] not in todo and len(todo) == len(PARTS) - 1
