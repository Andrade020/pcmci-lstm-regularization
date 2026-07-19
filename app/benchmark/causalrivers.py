"""
CausalRivers loader — a REAL multivariate time series with a publicly known
ground-truth causal graph.

CausalRivers (Stein et al. 2025, https://causalrivers.github.io/) publishes
river-discharge series from gauge stations together with a directed causal graph
in which an edge u -> v means station u is upstream of v, so u's discharge
physically causes v's a short travel time later. Direction is fixed by geography,
which makes it an honest real-world benchmark for causal discovery — and, because
all stations in a basin share the same rainfall, it also exhibits the exact
common-driver confounding this app now diagnoses.

We use the smallest region (the Elbe "flood" set: 42 stations, one connected
component) and carve out a small connected subnetwork with its matching series.
The full archive is ~127 MB and is cached (gitignored) under app/datasets/.

Ground-truth lag: the true travel-time lag between gauges is not published, so
each directed edge is mapped to lag 1 as a *structural* approximation (read the
resulting F1 as structural recovery, exactly like the Lorenz-96 / Kuramoto
entries — not an exact-lag score). The physically certain part is the direction.
"""
import os
import pickle
import urllib.request
import zipfile

import numpy as np
import pandas as pd

RIVERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "causalrivers")
PRODUCT_DIR = os.path.join(RIVERS_DIR, "product")
ZIP_URL = ("https://github.com/CausalRivers/benchmark/releases/download/"
           "First_release/product.zip")

_GRAPH_FILE = "rivers_flood.p"
_SERIES_FILE = "rivers_ts_flood.csv"
_META_FILE = "rivers_meta_flood.csv"


def _ensure_data():
    """Download + unzip the CausalRivers archive on first use (cached after)."""
    graph_path = os.path.join(PRODUCT_DIR, _GRAPH_FILE)
    if os.path.exists(graph_path):
        return
    os.makedirs(RIVERS_DIR, exist_ok=True)
    zip_path = os.path.join(RIVERS_DIR, "product.zip")
    try:
        print(f"[causalrivers] downloading {ZIP_URL} (~127 MB, one-time)…")
        urllib.request.urlretrieve(ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RIVERS_DIR)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not download the CausalRivers archive. It is ~127 MB from "
            f"{ZIP_URL}. Check your connection or fetch it manually into "
            f"{RIVERS_DIR}.") from exc
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def _pick_subgraph(G, n_vars, seed=0):
    """
    Deterministically pick a connected node set of size `n_vars` maximizing the
    number of induced directed edges (i.e. the richest small causal structure).
    BFS on the undirected view from each seed node; ties broken by node id.
    """
    import networkx as nx
    U = G.to_undirected()
    best_nodes, best_edges = None, -1
    for s in sorted(G.nodes()):
        seen = [s]
        frontier = [s]
        while len(seen) < n_vars and frontier:
            nxt = []
            for u in frontier:
                for w in sorted(U.neighbors(u)):
                    if w not in seen:
                        seen.append(w)
                        nxt.append(w)
                        if len(seen) >= n_vars:
                            break
                if len(seen) >= n_vars:
                    break
            frontier = nxt
        if len(seen) >= n_vars:
            nodes = seen[:n_vars]
            e = G.subgraph(nodes).number_of_edges()
            if e > best_edges:
                best_edges, best_nodes = e, list(nodes)
    if best_nodes is None:
        raise RuntimeError(f"No connected subgraph of size {n_vars} found.")
    return best_nodes


def load_causalrivers(n_vars: int = 6, resample: str = "1h", seed: int = 0,
                      difference: bool = True):
    """
    Load a small CausalRivers subnetwork as (data, var_names, true_links).

    Args:
        n_vars: number of gauge stations (connected subnetwork).
        resample: pandas offset to aggregate the 15-min series (default hourly).
        seed: reserved for tie-breaking / reproducibility.
        difference: first-difference each series (the flood-discharge levels are
            strongly non-stationary; the differenced series is what PCMCI and the
            LSTMs model, and it keeps the shared-rainfall co-movement intact).

    Returns:
        (data (T, N) float32, var_names list, true_links {j: [(i, 1)]}).
    """
    _ensure_data()
    import networkx as nx  # noqa: F401  (import validated here for a clear error)

    G = pickle.load(open(os.path.join(PRODUCT_DIR, _GRAPH_FILE), "rb"))
    df = pd.read_csv(os.path.join(PRODUCT_DIR, _SERIES_FILE),
                     index_col=0, parse_dates=True)
    meta = pd.read_csv(os.path.join(PRODUCT_DIR, _META_FILE), index_col=0)

    nodes = _pick_subgraph(G, n_vars, seed)
    cols = [str(n) for n in nodes]
    sub = df[cols].copy()

    # aggregate to a coarser step, then fill the few gauge gaps
    sub = sub.resample(resample).mean()
    sub = sub.interpolate(limit_direction="both").dropna()

    data = sub.values.astype(np.float32)
    if difference:
        data = np.diff(data, axis=0)

    # readable, unique labels: "River#id"
    var_names = []
    for n in nodes:
        river = str(meta.loc[n, "R"]) if n in meta.index else ""
        river = river.strip() or "st"
        var_names.append(f"{river}#{n}")

    # ground-truth directed links, mapped to lag 1 (structural approximation)
    idx = {n: k for k, n in enumerate(nodes)}
    true_links = {k: [] for k in range(len(nodes))}
    for u, v in G.subgraph(nodes).edges():
        if u in idx and v in idx and u != v:
            true_links[idx[v]].append((idx[u], 1))   # u (upstream) -> v (downstream)

    return data, var_names, true_links
