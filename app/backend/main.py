"""
FastAPI backend for the causal forecasting app.

Endpoints
    GET  /                     -> the single-page front-end
    GET  /api/datasets         -> example datasets available in the registry
    POST /api/run/example      -> run pipeline on a named example dataset
    POST /api/run/upload       -> run pipeline on an uploaded wide CSV
    GET  /api/jobs/{id}        -> job status (poll this)
    GET  /api/jobs/{id}/result -> metrics table + DM tests + base64 figures

Long runs execute in a background thread (see jobs.py); the UI polls status.
"""
import io
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.data import stationarize
from .figures import build_all_figures
from .jobs import JobRegistry
from .pipeline import run_pipeline
from .schemas import RunConfig, RunExampleRequest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(APP_DIR, "frontend")

app = FastAPI(title="Causal Forecasting App")
registry = JobRegistry()


# ── Dataset registry (optional; app still works with uploads if it fails) ──────

def _load_registry():
    try:
        from app.benchmark import datasets as ds
        return ds
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] dataset registry unavailable: {exc}")
        return None


# ── Job execution ──────────────────────────────────────────────────────────────

def _make_target(data, var_names, cfg, true_links):
    def target(job):
        def progress(msg, frac):
            job.message = msg
            job.progress = float(frac)
        result = run_pipeline(data, var_names, cfg,
                              true_links=true_links, progress=progress)
        job.message = "rendering figures"
        job.figures = build_all_figures(result)
        # strip the heavy arrays before storing for JSON responses
        result.pop("_arrays", None)
        result.pop("history", None)
        job.result = result
    return target


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets():
    ds = _load_registry()
    if ds is None:
        return {"datasets": []}
    return {"datasets": ds.list_datasets()}


@app.post("/api/run/example")
def run_example(req: RunExampleRequest):
    ds = _load_registry()
    if ds is None:
        raise HTTPException(500, "Dataset registry unavailable")
    try:
        data, var_names, true_links = ds.load_dataset(req.dataset)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset '{req.dataset}'")

    cfg = req.config.to_pipeline_cfg()
    job = registry.create({"dataset": req.dataset, "config": cfg})
    registry.run_async(job, _make_target(data, var_names, cfg, true_links))
    return {"job_id": job.id}


@app.post("/api/run/upload")
async def run_upload(
    file: UploadFile = File(...),
    window: int = Form(12), tau_max: int = Form(6),
    alpha: float = Form(0.05), pc_alpha: float = Form(0.1),
    hidden: int = Form(64), num_layers: int = Form(2),
    epochs: int = Form(150), lr: float = Form(5e-4), batch: int = Form(32),
    transform: str = Form(""),
):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = stationarize(df, transform or None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    var_names = list(df.columns)
    data = df.values.astype(np.float32)
    if data.shape[1] < 2:
        raise HTTPException(400, "Need at least 2 numeric series columns.")

    cfg = RunConfig(window=window, tau_max=tau_max, alpha=alpha, pc_alpha=pc_alpha,
                    hidden=hidden, num_layers=num_layers, epochs=epochs,
                    lr=lr, batch=batch).to_pipeline_cfg()
    job = registry.create({"dataset": file.filename, "config": cfg,
                           "shape": list(data.shape)})
    registry.run_async(job, _make_target(data, var_names, cfg, None))
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    return job.public()


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    if job.status != "done":
        raise HTTPException(409, f"Job not finished (status={job.status})")
    return JSONResponse({"result": job.result, "figures": job.figures})


# ── Front-end (mounted last so /api/* wins) ────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
