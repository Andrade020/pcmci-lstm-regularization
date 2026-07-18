"""
Climate teleconnection indices.

The repo already ships 7 monthly NOAA indices under `data/climate/`
(NINO34, PDO, TSA, TNA, NAO, AO, PNA). This module loads that bundled panel and,
when the network is available, tries to enrich it with a few additional indices
from NOAA PSL (SOI, AMO, DMI/IOD). Everything falls back gracefully to the
bundled panel so the benchmark always runs offline.

Climate has no exact causal ground truth, so these datasets are used for
forecast-accuracy comparison (DM tests) only, not graph-recovery F1.
"""
import os

import numpy as np
import pandas as pd

CLIMATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "climate")

# Extra NOAA PSL indices to try (standard correlation ".data" text format).
_PSL_EXTRA = {
    "SOI": "https://psl.noaa.gov/data/correlation/soi.data",
    "AMO": "https://psl.noaa.gov/data/correlation/amon.us.data",
    "DMI": "https://psl.noaa.gov/data/correlation/dmi.had.long.data",
}


def load_climate_bundled() -> tuple:
    """Load and align the 7 bundled monthly indices. Returns (df, array, names)."""
    frames = {}
    for fname in sorted(os.listdir(CLIMATE_DIR)):
        if not fname.endswith(".csv"):
            continue
        name = fname[:-4]
        s = pd.read_csv(os.path.join(CLIMATE_DIR, fname),
                        index_col=0, parse_dates=True)
        frames[name] = s.iloc[:, 0]
    panel = pd.DataFrame(frames).dropna()
    panel = panel.loc["1950-01-01":"2024-12-01"]
    return panel, panel.values.astype(np.float32), list(panel.columns)


def _parse_psl(text: str, name: str) -> pd.Series:
    """Parse a NOAA PSL correlation '.data' text file into a monthly series."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header = lines[0].split()
    start_year, end_year = int(header[0]), int(header[1])
    missing = None
    records = {}
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) == 1:
            missing = float(parts[0])
            continue
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if year < start_year or year > end_year:
            continue
        for m in range(12):
            val = float(parts[1 + m])
            records[pd.Timestamp(year=year, month=m + 1, day=1)] = val
    s = pd.Series(records, name=name).sort_index()
    if missing is not None:
        s = s.replace(missing, np.nan)
    return s


def download_extra_indices(cache_dir: str, timeout: int = 15) -> dict:
    """Try to download extra PSL indices; return {name: series}. Empty on failure."""
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for name, url in _PSL_EXTRA.items():
        cache_file = os.path.join(cache_dir, f"{name}.csv")
        try:
            if os.path.exists(cache_file):
                s = pd.read_csv(cache_file, index_col=0, parse_dates=True).iloc[:, 0]
            else:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    text = resp.read().decode("utf-8", errors="ignore")
                s = _parse_psl(text, name).dropna()
                s.to_frame(name).to_csv(cache_file)
            out[name] = s
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not fetch {name}: {exc}")
    return out


def load_climate_extended(cache_dir: str) -> tuple:
    """Bundled 7 indices + any downloadable extras, aligned. Returns (df, array, names)."""
    panel, _, _ = load_climate_bundled()
    extras = download_extra_indices(cache_dir)
    if extras:
        merged = panel.copy()
        for name, s in extras.items():
            merged[name] = s
        merged = merged.dropna()
        if len(merged) >= 300:   # only keep extras if enough overlap survives
            panel = merged
    return panel, panel.values.astype(np.float32), list(panel.columns)
