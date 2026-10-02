"""Run a pipeline Spark module on Dataproc Serverless: the same code as local, data in gs://$GCP_BUCKET.

    uv run python -m ctrisk.cloud.dataproc ctrisk.spark.clean_trials [--dry-run]

Builds the ctrisk wheel, ships it with the pure-Python fallback of ijson (the FAERS parser runs on the
executors), submits a batch capped at MAX_EXECUTORS with a TTL, and prints a worst-case cost first.
Dataproc Serverless has no internet by default; none of the Spark jobs needs it.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import ijson

REPO = Path(__file__).resolve().parents[3]
RUNTIME = "2.2"                    # Spark 3.5, like the local pyspark pin
DCU_PER_NODE = 4                   # driver and each executor: 4 vCPU = 4 DCUs
USD_PER_DCU_HOUR = 0.06            # Dataproc Serverless standard compute, us-central1 list price
PASSED_ENV = ("MIN_TRIALS", "MIN_MATCH_RATE", "FAERS_YEARS", "FAERS_QUARTERS")
ENTRY = "import runpy, sys\nsys.argv = sys.argv[1:]\nrunpy.run_module(sys.argv[0], run_name='__main__')\n"


def estimate_usd(max_executors: int, minutes: float) -> float:
    """Upper bound: every executor busy for the whole run."""
    return round(DCU_PER_NODE * (1 + max_executors) * minutes / 60 * USD_PER_DCU_HOUR, 2)


def batch_command(module: str, bucket: str, region: str, code: list[str], entry: str, env: dict,
                  batch_id: str, max_executors: int = 8, ttl: str = "2h") -> list[str]:
    props = [f"spark.dynamicAllocation.maxExecutors={max_executors}", "spark.executor.cores=4",
             "spark.driver.cores=4"]
    for k, v in env.items():
        props += [f"spark.dataproc.driverEnv.{k}={v}", f"spark.executorEnv.{k}={v}"]
    return ["gcloud", "dataproc", "batches", "submit", "pyspark", entry,
            f"--region={region}", f"--batch={batch_id}", f"--version={RUNTIME}", f"--ttl={ttl}",
            f"--py-files={','.join(code)}", f"--properties=^#^{'#'.join(props)}",
            "--", module]


def _deps_zip(folder: Path) -> Path:
    """ijson without its compiled backends: on Linux it falls back to its pure-Python parser."""
    out = folder / "deps.zip"
    root = Path(ijson.__file__).parent
    with zipfile.ZipFile(out, "w") as z:
        for f in root.rglob("*.py"):
            z.write(f, Path("ijson") / f.relative_to(root))
    return out


def upload_code(bucket: str) -> tuple[list[str], str]:
    folder = Path(tempfile.mkdtemp())
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(folder)], cwd=REPO, check=True,
                   capture_output=True)
    wheel = next(folder.glob("ctrisk-*.whl"))
    shutil.copy(wheel, folder / "ctrisk.zip")             # --py-files wants a zip; a wheel is one
    (folder / "run_module.py").write_text(ENTRY)
    files = [folder / "ctrisk.zip", _deps_zip(folder), folder / "run_module.py"]
    subprocess.run(["gcloud", "storage", "cp", *map(str, files), f"gs://{bucket}/code/"], check=True)
    return [f"gs://{bucket}/code/ctrisk.zip", f"gs://{bucket}/code/deps.zip"], f"gs://{bucket}/code/run_module.py"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if len(args) != 1:
        sys.exit(__doc__)
    module = args[0]
    bucket, region = os.environ["GCP_BUCKET"], os.getenv("GCP_REGION", "us-central1")
    max_executors = int(os.getenv("DATAPROC_MAX_EXECUTORS", "8"))
    env = {"MODE": "cloud", "GCP_BUCKET": bucket, **{k: os.environ[k] for k in PASSED_ENV if os.getenv(k)}}
    batch_id = f"ctrisk-{module.rsplit('.', 1)[-1].replace('_', '-')}-{int(time.time())}"
    print(f"{module} on Dataproc Serverless, {region}, up to {max_executors} executors: at most "
          f"${estimate_usd(max_executors, 30)} per 30 minutes")
    code, entry = ((["gs://BUCKET/code/ctrisk.zip", "gs://BUCKET/code/deps.zip"], "gs://BUCKET/code/run_module.py")
                   if dry else upload_code(bucket))
    cmd = batch_command(module, bucket, region, code, entry, env, batch_id, max_executors)
    print(" ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)
