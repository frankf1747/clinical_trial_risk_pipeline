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
