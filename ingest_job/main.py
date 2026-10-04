"""Cloud Run Job: copy every FAERS drug/event file listed by openFDA into gs://$BUCKET/raw/faers_full/<quarter>/.

The full history is ~110 GB of zips, too much to send from a laptop; Cloud Run downloads it inside Google's
network. Each of the job's parallel tasks takes every Nth file (CLOUD_RUN_TASK_INDEX of CLOUD_RUN_TASK_COUNT),
streams it from openFDA into GCS without touching disk, and skips files already there at the listed size,
so a rerun only fetches what is missing. Same <quarter>/<file> layout as the laptop download, under
raw/faers_full/ so one openFDA release is never mixed with files downloaded earlier.
"""
import os
import sys
import time

import requests

INDEX_URL = "https://api.fda.gov/download.json"
MB = 2**20


def object_name(url: str, prefix: str) -> str:
    quarter, name = url.split("/")[-2:]
    return f"{prefix}/{quarter}/{name}"


def share(partitions: list[dict], task: int, tasks: int) -> list[dict]:
    return partitions[task::tasks]


def missing(partitions: list[dict], prefix: str, listed: set[str]) -> list[dict]:
    """Files not yet in GCS. An upload through blob.open() creates the object only when it completes, so an
    object that exists is whole. (openFDA's size_mb is approximate and its files are regenerated over time,
    so sizes cannot be compared.)"""
    return [p for p in partitions if object_name(p["file"], prefix) not in listed]


def copy(url: str, blob, attempts: int = 4) -> int:
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with blob.open("wb", chunk_size=16 * MB, content_type="application/zip") as out:
                    n = 0
                    for chunk in r.iter_content(4 * MB):
                        out.write(chunk)
                        n += len(chunk)
            return n
        except (requests.RequestException, OSError) as e:
            if attempt == attempts:
                raise
            print(f"retry {attempt} for {url}: {e}", flush=True)
            time.sleep(10 * attempt)
    return 0


if __name__ == "__main__":
    from google.cloud import storage

    bucket_name, prefix = os.environ["BUCKET"], os.getenv("PREFIX", "raw/faers_full")
    task, tasks = int(os.getenv("CLOUD_RUN_TASK_INDEX", "0")), int(os.getenv("CLOUD_RUN_TASK_COUNT", "1"))
    partitions = requests.get(INDEX_URL, timeout=60).json()["results"]["drug"]["event"]["partitions"]
    partitions = sorted(partitions, key=lambda p: p["file"])
    bucket = storage.Client().bucket(bucket_name)
    listed = {b.name for b in bucket.list_blobs(prefix=prefix + "/")}
    todo = missing(share(partitions, task, tasks), prefix, listed)
    print(f"task {task}/{tasks}: {len(todo)} files to copy", flush=True)
    copied = 0
    for i, p in enumerate(todo, 1):
        copied += copy(p["file"], bucket.blob(object_name(p["file"], prefix)))
        if i % 10 == 0 or i == len(todo):
            print(f"task {task}: {i}/{len(todo)} files, {copied / 2**30:.1f} GB", flush=True)
    sys.exit(0)
