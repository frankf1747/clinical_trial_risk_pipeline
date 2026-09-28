"""Model versions on disk: models/v{N}/ holds model.pkl plus one JSON file per document."""
import json
import re
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def _versions(root: Path) -> list[int]:
    return sorted(int(m[1]) for p in Path(root).glob("v*") if (m := re.fullmatch(r"v(\d+)", p.name)))


def next_version(root: Path = MODELS_DIR) -> int:
    return (_versions(root) or [0])[-1] + 1


def latest(root: Path = MODELS_DIR) -> int:
    if not _versions(root):
        raise FileNotFoundError(f"no model versions in {root}; run `make train` first")
    return _versions(root)[-1]


def save(root: Path, version: int, model, **docs: dict) -> Path:
    folder = Path(root) / f"v{version}"
    folder.mkdir(parents=True, exist_ok=False)          # never overwrite a version
    joblib.dump(model, folder / "model.pkl")
    for name, doc in docs.items():
        (folder / f"{name}.json").write_text(json.dumps(doc, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return folder


def load(root: Path, version: int):
    folder = Path(root) / f"v{version}"
    docs = {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}
    return joblib.load(folder / "model.pkl"), docs
