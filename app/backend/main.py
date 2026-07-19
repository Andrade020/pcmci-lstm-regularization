"""
FastAPI backend for the causal forecasting app.

Serves JSON (not rendered images) so the front-end can draw everything with
interactive SVG: the imported series, the PCMCI causal graph, and the forecasts.

Endpoints
    GET  /                          -> the single-page front-end
    GET  /api/datasets              -> example datasets in the registry
    GET  /api/dataset/{name}/preview-> raw series of an example (for the preview chart)
    POST /api/preview               -> raw series of an uploaded CSV (preview)
    GET  /api/example.csv           -> a downloadable example CSV
    POST /api/run/example           -> run the pipeline on a named example dataset
    POST /api/run/upload            -> run the pipeline on an uploaded wide CSV
    GET  /api/jobs/{id}             -> job status (poll this)
    GET  /api/jobs/{id}/result      -> metrics + causal graph + forecast series (JSON)

Long runs execute in a background thread (see jobs.py); the UI polls status.
"""
import io
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from src.data import stationarize
from .jobs import JobRegistry
from .pipeline import run_pipeline
from .schemas import RunConfig, RunExampleRequest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(APP_DIR, "frontend")

app = FastAPI(title="Causal Forecasting App")
registry = JobRegistry()

# Models shown in the forecast chart (kept small for legibility).
FORECAST_MODELS = ["LSTM Baseline", "LSTM Causal (PCMCI)",
                   "LSTM Masked (PCMCI)", "VAR"]


# ── Dataset registry (optional; app still works with uploads if it fails) ──────

def _load_registry():
    try:
        from app.benchmark import datasets as ds
        return ds
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] dataset registry unavailable: {exc}")
        return None


# ── Serialization helpers ──────────────────────────────────────────────────────

def _downsample_series(arr, var_names, max_points=500):
    """Return a JSON-friendly, downsampled view of a (T, N) array for charts."""
    arr = np.asarray(arr, dtype=float)
    T = arr.shape[0]
    step = max(1, T // max_points)
    idx = list(range(0, T, step))
    values = {var_names[j]: [round(float(arr[i, j]), 6) for i in idx]
              for j in range(arr.shape[1])}
    return {"var_names": list(var_names), "index": idx,
            "n_obs": int(T), "n_vars": int(arr.shape[1]), "values": values}


def _build_forecast(result):
    """Per-variable truth + selected model predictions over the test window."""
    arrays = result["_arrays"]
    y_true = np.asarray(arrays["y_true"], dtype=float)
    preds = arrays["preds"]
    var_names = result["var_names"]
    T = y_true.shape[0]

    def col(a, j):
        return [round(float(a[t, j]), 6) for t in range(T)]

    truth = {vn: col(y_true, j) for j, vn in enumerate(var_names)}
    models = {}
    for name in FORECAST_MODELS:
        if name in preds:
            arr = np.asarray(preds[name], dtype=float)
            models[name] = {vn: col(arr, j) for j, vn in enumerate(var_names)}
    return {"var_names": var_names, "index": list(range(T)),
            "truth": truth, "models": models}


def _build_verdict(result):
    """Plain-language, JSON verdict: is there causal structure, and does it help?"""
    m = result["models"]
    n_links = result["n_links"]
    best = min(m.items(), key=lambda kv: kv[1]["mae"])[0]

    def dm(name):
        return m[name] if name in m else None

    soft = dm("LSTM Causal (PCMCI)")
    masked = dm("LSTM Masked (PCMCI)")

    def beats_baseline(entry):
        if not entry or entry.get("dm_stat") is None:
            return None
        return bool(entry["dm_stat"] > 0 and entry["p_value"] < 0.05)

    # graph informative vs random (lower MSE than the density-matched random graph)
    graph_informative = None
    if "LSTM Causal (PCMCI)" in m and "LSTM Causal (Random)" in m:
        graph_informative = bool(m["LSTM Causal (PCMCI)"]["mse"]
                                 < m["LSTM Causal (Random)"]["mse"])

    diag = result.get("diagnostics", {}) or {}
    return {
        "n_links": int(n_links),
        "has_structure": bool(n_links > 0),
        "graph_f1": None if result["graph_f1"] is None else result["graph_f1"],
        "best_model": best,
        "soft_beats_baseline": beats_baseline(soft),
        "masked_beats_baseline": beats_baseline(masked),
        "graph_informative": graph_informative,
        "method": result.get("method", "pcmci"),
        "deseason_period": result.get("deseason_period"),
        "confounding_suspected": bool(diag.get("confounding_suspected")),
        "confounding_reasons": diag.get("reasons", []),
        "graph_density": diag.get("graph_density"),
        "seasonality": diag.get("seasonality"),
    }


# ── Job execution ──────────────────────────────────────────────────────────────

def _make_target(data, var_names, cfg, true_links):
    def target(job):
        def progress(msg, frac):
            job.message = msg
            job.progress = float(frac)
        result = run_pipeline(data, var_names, cfg,
                              true_links=true_links, progress=progress)
        job.message = "packaging results"
        payload = {k: v for k, v in result.items()
                   if k not in ("_arrays", "history")}
        payload["forecast"] = _build_forecast(result)
        payload["verdict"] = _build_verdict(result)
        job.result = payload
    return target


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets():
    ds = _load_registry()
    if ds is None:
        return {"datasets": []}
    return {"datasets": ds.list_datasets()}


@app.get("/api/dataset/{name}/preview")
def dataset_preview(name: str):
    ds = _load_registry()
    if ds is None:
        raise HTTPException(500, "Dataset registry unavailable")
    try:
        data, var_names, _ = ds.load_dataset(name)
    except KeyError:
        raise HTTPException(404, f"Unknown dataset '{name}'")
    out = _downsample_series(data, var_names)
    meta = {d["name"]: d for d in ds.list_datasets()}.get(name, {})
    out["description"] = meta.get("description", "")
    out["group"] = meta.get("group", "")
    return out


@app.post("/api/preview")
async def upload_preview(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse CSV: {exc}")
    if df.shape[1] < 2:
        raise HTTPException(400, "Need at least 2 numeric series columns.")
    return _downsample_series(df.values, list(df.columns))


@app.get("/api/example.csv")
def example_csv():
    """A small self-contained wide CSV so users can try the upload flow."""
    ds = _load_registry()
    if ds is None:
        raise HTTPException(500, "Dataset registry unavailable")
    data, var_names, _ = ds.load_dataset("var4")
    df = pd.DataFrame(np.asarray(data)[:300], columns=var_names)
    df.insert(0, "t", range(len(df)))
    return PlainTextResponse(df.to_csv(index=False),
                             media_type="text/csv",
                             headers={"Content-Disposition":
                                      "attachment; filename=example_var4.csv"})


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
    method: str = Form("pcmci"), deseason_period: int = Form(0),
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
                    lr=lr, batch=batch, method=method or "pcmci",
                    deseason_period=(deseason_period or None)).to_pipeline_cfg()
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
    return JSONResponse({"result": job.result})


# ── Front-end ──────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/app.js")
def app_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "app.js"))


@app.get("/style.css")
def style_css():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"))
