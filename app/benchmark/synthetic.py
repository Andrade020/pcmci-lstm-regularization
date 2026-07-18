"""
Synthetic systems with known (or structurally known) causal ground truth.

Linear VAR systems (var4, var8) and the nonlinear threshold system are imported
from `experiments.exp_synthetic` so the benchmark uses the exact same generators
and ground-truth links as the paper. Here we add two chaotic dynamical systems
commonly used as causal-discovery benchmarks:

  * Lorenz-96 — a spatially extended chaotic system on a ring; the equation
    dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F gives each node the
    structural parents {i-2, i-1, i+1} (plus the self term i).
  * Kuramoto — coupled phase oscillators on a sparse graph; each oscillator's
    parents are its neighbours in the coupling matrix.

For the chaotic systems the "ground truth" is the *coupling topology* mapped to
lag 1. This is an approximation — a continuous chaotic flow has no exact
discrete-lag DAG — so F1 against it should be read as structural recovery, not
an exact score. This caveat is surfaced in the benchmark report.
"""
import numpy as np

# Reuse the paper's exact generators + ground-truth link dicts.
from experiments.exp_synthetic import (
    generate_4var, TRUE_LINKS_4VAR,
    A_8VAR, TRUE_LINKS_8VAR,
    generate_threshold_4var, TRUE_LINKS_THRESHOLD,
)
from src.data import generate_var_series


# ── Lorenz-96 ────────────────────────────────────────────────────────────────

def generate_lorenz96(N=6, T=2000, F=8.0, dt=0.05, sample_every=2,
                      burn_in=500, noise_std=0.0, seed=42):
    """
    Integrate the Lorenz-96 system with RK4 and return a sampled trajectory.

    Returns (T, N) array. Structural parents of node i: {i-2, i-1, i+1}.
    """
    rng = np.random.default_rng(seed)

    def deriv(x):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

    x = F * np.ones(N) + 0.01 * rng.standard_normal(N)
    steps = burn_in + T * sample_every
    out = np.zeros((T, N))
    k = 0
    for s in range(steps):
        k1 = deriv(x)
        k2 = deriv(x + 0.5 * dt * k1)
        k3 = deriv(x + 0.5 * dt * k2)
        k4 = deriv(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if noise_std > 0:
            x = x + noise_std * rng.standard_normal(N)
        if s >= burn_in and (s - burn_in) % sample_every == 0 and k < T:
            out[k] = x
            k += 1
    return out


def lorenz96_true_links(N):
    """Structural parents {i-2, i-1, i+1} at lag 1 (ring topology)."""
    links = {j: [] for j in range(N)}
    for j in range(N):
        for off in (-2, -1, 1):
            i = (j + off) % N
            links[j].append((i, 1))
        links[j].append((j, 1))  # self term
    return links


# ── Kuramoto ─────────────────────────────────────────────────────────────────

def generate_kuramoto(N=5, T=2000, K=2.0, dt=0.05, sample_every=2,
                      burn_in=500, seed=42):
    """
    Coupled phase oscillators on a ring; observed signal is sin(theta_i).

    dtheta_i/dt = omega_i + (K/deg_i) * sum_j A_ij sin(theta_j - theta_i)
    Returns (T, N) array of sin(theta). Parents of i: its ring neighbours.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i - 1) % N] = 1
        A[i, (i + 1) % N] = 1
    deg = A.sum(axis=1)
    omega = rng.uniform(0.8, 1.2, size=N)
    theta = rng.uniform(0, 2 * np.pi, size=N)

    def deriv(th):
        diff = th[None, :] - th[:, None]        # theta_j - theta_i
        return omega + (K / deg) * (A * np.sin(diff)).sum(axis=1)

    steps = burn_in + T * sample_every
    out = np.zeros((T, N))
    k = 0
    for s in range(steps):
        k1 = deriv(theta)
        k2 = deriv(theta + 0.5 * dt * k1)
        k3 = deriv(theta + 0.5 * dt * k2)
        k4 = deriv(theta + dt * k3)
        theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if s >= burn_in and (s - burn_in) % sample_every == 0 and k < T:
            out[k] = np.sin(theta)
            k += 1
    return out, A


def kuramoto_true_links(A):
    """Parents of i = ring neighbours (nonzero A rows), at lag 1, plus self."""
    N = A.shape[0]
    links = {j: [] for j in range(N)}
    for j in range(N):
        for i in range(N):
            if A[j, i] != 0:
                links[j].append((i, 1))
        links[j].append((j, 1))
    return links
