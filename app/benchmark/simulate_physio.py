"""
NetSim-style physiological connectivity simulation.

Smith et al. (2011) "Network modelling methods for FMRI" simulate BOLD signals
from a known directed effective-connectivity network. Downloading the original
FMRIB sim files is not always possible offline, so this module provides a
self-contained, reproducible stand-in: a sparse first-order MVAR whose lag-1
coefficient matrix encodes a *known* directed connectivity graph, optionally
smoothed by a short haemodynamic-like kernel to mimic BOLD autocorrelation.

The ground-truth causal graph is exact (the nonzero off-diagonal lag-1 edges),
so graph-recovery F1 is meaningful here.
"""
import numpy as np


# A fixed 5-node directed effective-connectivity network (source -> target).
# Chosen to include a fork (0->1, 0->2), a chain (1->3), and a collider (2,3->4).
_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
_N = 5


def _connectivity_matrix(N=_N, edges=_EDGES, self_coef=0.5, edge_coef=0.4):
    """Build the (N, N) lag-1 coefficient matrix B with B[target, source]."""
    B = np.eye(N) * self_coef
    for src, tgt in edges:
        B[tgt, src] = edge_coef
    return B


def generate_netsim(T=1200, hrf_smooth=True, noise_std=0.3, seed=42):
    """
    Simulate the MVAR(1) network x_t = B x_{t-1} + eps, optional HRF smoothing.

    Returns (T, N) array. Ground truth from `netsim_true_links`.
    """
    rng = np.random.default_rng(seed)
    B = _connectivity_matrix()
    N = B.shape[0]
    burn = 200
    x = np.zeros(N)
    out = np.zeros((burn + T, N))
    for t in range(1, burn + T):
        x = B @ x + noise_std * rng.standard_normal(N)
        out[t] = x
    series = out[burn:]

    if hrf_smooth:
        # Short causal smoothing kernel (a crude HRF proxy): weighted trailing avg.
        kernel = np.array([0.6, 0.3, 0.1])
        smoothed = np.zeros_like(series)
        for i in range(N):
            smoothed[:, i] = np.convolve(series[:, i], kernel, mode="same")
        series = smoothed
    return series


def netsim_true_links(N=_N, edges=_EDGES):
    """{target: [(source, 1), ...]} including the self term at lag 1."""
    links = {j: [] for j in range(N)}
    for src, tgt in edges:
        links[tgt].append((src, 1))
    for j in range(N):
        links[j].append((j, 1))
    return links
