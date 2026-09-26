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
        bucket = os.getenv("GCP_BUCKET")
        if not bucket:
            raise ValueError("GCP_BUCKET is required when MODE=cloud")
        data_root = f"gs://{bucket}"
    else:
        raise ValueError(f"MODE must be 'local' or 'cloud', got {mode!r}")
    return Config(mode, local_root, data_root, int(os.getenv("MIN_TRIALS", "40000")))
