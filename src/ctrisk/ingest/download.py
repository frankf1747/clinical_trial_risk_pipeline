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
