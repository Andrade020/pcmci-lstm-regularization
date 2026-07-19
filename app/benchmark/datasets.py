"""
Dataset registry shared by the FastAPI app and the benchmark harness.

Each dataset exposes a loader returning (data (T,N) float32, var_names,
true_links|None) and a recommended pipeline config. Synthetic and physiological
systems carry an exact/structural ground-truth graph (enabling F1); climate does
not (forecast DM only). Generated/downloaded data is cached under the gitignored
`app/datasets/` directory.
"""
import os

import numpy as np

from src.data import generate_var_series
from .synthetic import (
    generate_4var, TRUE_LINKS_4VAR,
    A_8VAR, TRUE_LINKS_8VAR,
    generate_threshold_4var, TRUE_LINKS_THRESHOLD,
    generate_lorenz96, lorenz96_true_links,
    generate_kuramoto, kuramoto_true_links,
)
from .simulate_physio import generate_netsim, netsim_true_links
from .download_climate import load_climate_bundled, load_climate_extended
from .causalrivers import load_causalrivers

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "datasets")
os.makedirs(CACHE_DIR, exist_ok=True)

T_SYNTH = 2000


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_var4():
    data = generate_4var(T_SYNTH)
    return data, [f"X{i}" for i in range(4)], TRUE_LINKS_4VAR


def _load_var8():
    data = generate_var_series(A_8VAR, T_SYNTH, noise_std=0.3, seed=42)
    return data, [f"X{i}" for i in range(8)], TRUE_LINKS_8VAR


def _load_threshold4():
    data = generate_threshold_4var(T_SYNTH)
    return data, [f"X{i}" for i in range(4)], TRUE_LINKS_THRESHOLD


def _load_lorenz96():
    N = 6
    data = generate_lorenz96(N=N, T=T_SYNTH)
    return data, [f"L{i}" for i in range(N)], lorenz96_true_links(N)


def _load_kuramoto():
    data, A = generate_kuramoto(N=5, T=T_SYNTH)
    return data, [f"O{i}" for i in range(5)], kuramoto_true_links(A)


def _load_netsim():
    data = generate_netsim(T=1200)
    return data, [f"R{i}" for i in range(5)], netsim_true_links()


def _load_rivers():
    return load_causalrivers(n_vars=6)


def _load_climate():
    _, arr, names = load_climate_bundled()
    return arr, names, None


def _load_climate_ext():
    _, arr, names = load_climate_extended(CACHE_DIR)
    return arr, names, None


# ── Registry ─────────────────────────────────────────────────────────────────

DATASETS = {
    "var4": {
        "loader": _load_var4,
        "description": "Linear VAR, 4 vars, multi-lag — favourable regime (F1≈0.84).",
        "group": "synthetic",
        "cfg": {"window": 5, "tau_max": 5, "hidden": 16, "epochs": 80, "lr": 1e-3,
                "batch": 64, "train_frac": 0.60, "val_frac": 0.20},
    },
    "var8": {
        "loader": _load_var8,
        "description": "Linear VAR(1), 8 vars — hard regime, PCMCI F1≈0.36.",
        "group": "synthetic",
        "cfg": {"window": 5, "tau_max": 5, "hidden": 32, "epochs": 80, "lr": 1e-3,
                "batch": 64, "train_frac": 0.60, "val_frac": 0.20},
    },
    "threshold4": {
        "loader": _load_threshold4,
        "description": "Nonlinear threshold VAR, 4 vars — same structure as var4.",
        "group": "synthetic",
        "cfg": {"window": 5, "tau_max": 5, "hidden": 16, "epochs": 80, "lr": 1e-3,
                "batch": 64, "train_frac": 0.60, "val_frac": 0.20},
    },
    "lorenz96": {
        "loader": _load_lorenz96,
        "description": "Chaotic Lorenz-96 ring, 6 nodes — structural ground truth.",
        "group": "chaotic",
        "cfg": {"window": 5, "tau_max": 5, "hidden": 32, "epochs": 80, "lr": 1e-3,
                "batch": 64, "train_frac": 0.60, "val_frac": 0.20},
    },
    "kuramoto": {
        "loader": _load_kuramoto,
        "description": "Coupled Kuramoto oscillators, 5 nodes — structural truth.",
        "group": "chaotic",
        "cfg": {"window": 5, "tau_max": 5, "hidden": 32, "epochs": 80, "lr": 1e-3,
                "batch": 64, "train_frac": 0.60, "val_frac": 0.20},
    },
    "netsim": {
        "loader": _load_netsim,
        "description": "NetSim-style fMRI effective connectivity, 5 nodes — exact DAG.",
        "group": "physiology",
        "cfg": {"window": 5, "tau_max": 5, "hidden": 24, "epochs": 80, "lr": 1e-3,
                "batch": 64, "train_frac": 0.60, "val_frac": 0.20},
    },
    "rivers": {
        "loader": _load_rivers,
        "description": "CausalRivers: 6 real Elbe/Jahna gauge stations — known "
                       "upstream→downstream graph (direction certain; lag approx).",
        "group": "rivers",
        "cfg": {"window": 6, "tau_max": 6, "hidden": 24, "epochs": 100, "lr": 1e-3,
                "batch": 64, "train_frac": 0.60, "val_frac": 0.20,
                "graph_lag_agnostic": True},
    },
    "climate": {
        "loader": _load_climate,
        "description": "7 NOAA teleconnection indices (bundled) — no exact graph.",
        "group": "climate",
        "cfg": {"window": 12, "tau_max": 12, "hidden": 64, "epochs": 120, "lr": 5e-4,
                "batch": 32, "train_frac": 0.70, "val_frac": 0.15,
                "lambda_causal": [0.01, 0.1, 1.0, 5.0]},
    },
    "climate_ext": {
        "loader": _load_climate_ext,
        "description": "Bundled indices + downloadable extras (SOI/AMO/DMI) — no graph.",
        "group": "climate",
        "cfg": {"window": 12, "tau_max": 12, "hidden": 64, "epochs": 120, "lr": 5e-4,
                "batch": 32, "train_frac": 0.70, "val_frac": 0.15,
                "lambda_causal": [0.01, 0.1, 1.0, 5.0]},
    },
}


def list_datasets() -> list:
    """Public metadata for the UI (no heavy data loaded)."""
    out = []
    for name, spec in DATASETS.items():
        out.append({
            "name": name,
            "description": spec["description"],
            "group": spec["group"],
            "has_ground_truth": spec["group"] != "climate",
            "cfg": spec["cfg"],
        })
    return out


def load_dataset(name: str) -> tuple:
    """Return (data, var_names, true_links) for a registered dataset."""
    if name not in DATASETS:
        raise KeyError(name)
    return DATASETS[name]["loader"]()


def get_config(name: str) -> dict:
    """Recommended pipeline config overrides for a dataset."""
    return dict(DATASETS[name]["cfg"])
