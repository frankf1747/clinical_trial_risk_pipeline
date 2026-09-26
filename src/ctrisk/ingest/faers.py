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
    r = requests.get(INDEX_URL, timeout=60)
    r.raise_for_status()
    partitions = r.json()["results"]["drug"]["event"]["partitions"]
    urls = select_partitions(partitions, cfg.faers_years, cfg.faers_quarters)
    if not urls:
        raise SystemExit("no FAERS files match FAERS_YEARS / FAERS_QUARTERS")
    root = Path(cfg.local_root) / "raw" / "faers"
    print(f"{len(urls)} FAERS files -> {root}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(lambda u: download(u, local_path(u, root)), urls), 1):
            if i % 25 == 0 or i == len(urls):
                print(f"{i}/{len(urls)}")
