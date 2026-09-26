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
