"""Public trial lookup: an NCT ID in, its termination-risk score, rank and main reasons out.

Reads what `make publish` wrote: current.json (which version, which file, the model's validated numbers)
and the lookup file Snowflake unloaded. LOOKUP_SOURCE is gs://<bucket>/serving on Cloud Run, or a local
folder for development. Everything is loaded once at start; requests never touch GCS or Snowflake.
"""
import json
import os
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")

SCORE_TYPES = {
    "forward": "Active trial: scored by the current model, trained on trials that started 2008–2014.",
    "held_out": "Finished trial that started in 2015 or later: the model never saw it during training.",
    "out_of_fold": ("Finished trial that started 2008–2014: scored by a model trained without any trial "
                    "from its start year, so its own outcome was never used."),
}
PHASES = {"PHASE1": "1", "PHASE1/PHASE2": "1/2", "PHASE2": "2", "PHASE2/PHASE3": "2/3", "PHASE3": "3"}
SPONSORS = {"INDUSTRY": "Industry", "OTHER": "Academic or other", "GOVERNMENT": "Government"}


class Store:
    """The published lookup, held in memory."""

    def __init__(self):
        self.summary: dict = {}
        self.trials: pd.DataFrame = pd.DataFrame()

    def load(self, source: str) -> None:
        folder = _fetch(source) if source.startswith("gs://") else Path(source)
        summary = json.loads((folder / "current.json").read_text())
        trials = pd.read_parquet(folder / summary["lookup_file"])
        if trials.empty:
            raise RuntimeError(f"{summary['lookup_file']} has no rows")
        trials.columns = trials.columns.str.lower()
        self.summary, self.trials = summary, trials.set_index("nct_id")

    def get(self, nct_id: str) -> dict | None:
        if nct_id not in self.trials.index:
            return None
        row = self.trials.loc[nct_id].to_dict()
        row["nct_id"] = nct_id
        row["reasons"] = json.loads(row.get("reasons") or "[]")
        row["start_date"] = None if pd.isna(row.get("start_date")) else str(row["start_date"])[:10]
        return row


def _fetch(source: str) -> Path:
    """Download current.json and the file it names from gs://bucket/prefix into a temp folder."""
    from google.cloud import storage                       # only needed on Cloud Run

    bucket_name, _, prefix = source.removeprefix("gs://").partition("/")
    bucket = storage.Client().bucket(bucket_name)
    folder = Path(tempfile.mkdtemp())
    bucket.blob(f"{prefix}/current.json").download_to_filename(folder / "current.json")
    name = json.loads((folder / "current.json").read_text())["lookup_file"]
    bucket.blob(f"{prefix}/{name}").download_to_filename(folder / name)
    return folder


STORE = Store()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("LOOKUP_SOURCE"):
        STORE.load(os.environ["LOOKUP_SOURCE"])
    yield


app = FastAPI(title="Clinical trial termination risk", lifespan=lifespan, docs_url="/api/docs")


def normalize(raw: str) -> str:
    """' nct03801083 ' or '03801083' -> 'NCT03801083'."""
    digits = re.sub(r"\D", "", raw or "")
    return f"NCT{digits.zfill(8)}" if digits else ""


def _context(request: Request, **extra) -> dict:
    return {"request": request, "s": STORE.summary, "phases": PHASES, "sponsors": SPONSORS, **extra}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html", _context(request))


@app.get("/trial")
def lookup(nct: str = ""):
    return RedirectResponse(f"/trial/{normalize(nct)}" if normalize(nct) else "/", status_code=303)


@app.get("/trial/{nct_id}", response_class=HTMLResponse)
def trial(request: Request, nct_id: str):
    nct_id = normalize(nct_id)
    row = STORE.get(nct_id)
    if row is None:
        return templates.TemplateResponse(request, "not_found.html", _context(request, nct_id=nct_id),
                                          status_code=404)
    return templates.TemplateResponse(request, "trial.html", _context(
        request, t=row, how=SCORE_TYPES.get(row["score_type"], ""),
        biggest=max([abs(r["contribution"]) for r in row["reasons"]] or [1])))


@app.get("/api/trials/{nct_id}")
def api_trial(nct_id: str):
    row = STORE.get(normalize(nct_id))
    if row is None:
        return JSONResponse({"detail": "not in the modelled population or the published snapshot"}, status_code=404)
    return {**row, "model_version": row.get("model_version")}


@app.get("/healthz")
def health():
    return {"status": "ok", "trials": len(STORE.trials), "model_version": STORE.summary.get("model_version")}
