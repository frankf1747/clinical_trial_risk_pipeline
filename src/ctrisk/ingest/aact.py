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
    is_url = source.startswith(("http://", "https://"))
    zip_path = _download(source, dest.parent / "aact.zip") if is_url else Path(source)
    return _extract(zip_path, dest, tables)


def _download(url: str, target: Path) -> Path:
    if target.exists():
        print(f"skip download, {target} exists")
        return target
    partial = target.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(partial, "wb") as f:
            f.writelines(r.iter_content(1 << 20))
    if not zipfile.is_zipfile(partial):
        partial.unlink()
        raise ValueError("download is not a zip; download it in a browser and pass the local path")
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
