"""
Causal discovery utilities wrapping Tigramite's PCMCI.

Three discovery methods are exposed through `run_pcmci(..., method=...)`:

  - "pcmci"      — the original lagged-only PCMCI (fast, the default).
  - "pcmci_plus" — PCMCI+ (also handles contemporaneous links / autocorrelation).
  - "lpcmci"     — LPCMCI, built for *latent confounders*. When two series share
                   an unobserved common cause (e.g. a calendar/campaign driver),
                   LPCMCI marks that edge as bidirected ('<->') instead of forcing
                   a spurious direct link, and we keep only the genuine directed
                   ('-->') edges. This is the principled fix for the dense
                   "everything causes everything" graph produced by a common driver.
"""
import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr


def _make_cond_ind_test(test: str):
    if test == "parcorr":
        return ParCorr()
    if test == "robust_parcorr":
        return RobustParCorr()
    raise ValueError(f"Unknown test: {test}. Choose 'parcorr' or 'robust_parcorr'.")


def links_from_graph(graph: np.ndarray) -> dict:
    """
    Extract directed lagged links {j: [(i, lag), ...]} from a Tigramite graph array.

    `graph` is a (N, N, tau_max+1) array of edge strings. A directed lagged edge
    from X_i at t-tau to X_j at t is encoded as graph[i, j, tau] == '-->'. Only
    those are kept: bidirected ('<->', latent confounder) and contemporaneous
    (tau=0) marks are deliberately dropped, so the link set is the set of genuine
    directed causes usable as a forecasting mask.
    """
    N = graph.shape[0]
    tau_max = graph.shape[2] - 1
    links = {j: [] for j in range(N)}
    for j in range(N):
        for i in range(N):
            for tau in range(1, tau_max + 1):
                if graph[i, j, tau] == "-->":
                    links[j].append((i, tau))
    return links


def run_pcmci(
    data: np.ndarray,
    tau_max: int = 5,
    alpha: float = 0.05,
    pc_alpha: float = 0.1,
    test: str = "parcorr",
    var_names: list = None,
    method: str = "pcmci",
) -> dict:
    """
    Run a causal-discovery method on a multivariate time series and return the
    significant lagged causal links.

    Args:
        data: (T, N) normalized numpy array (should be stationary)
        tau_max: maximum lag to test
        alpha: significance threshold for final MCI test
        pc_alpha: significance threshold for PC1 phase (liberal)
        test: 'parcorr' or 'robust_parcorr'
        var_names: optional list of variable names for output
        method: 'pcmci' (default), 'pcmci_plus', or 'lpcmci'

    Returns:
        dict with keys:
          'links'     — {j: [(i, lag), ...]} dict of significant edges (lag > 0)
          'p_matrix'  — (N, N, tau_max+1) p-value matrix
          'val_matrix'— (N, N, tau_max+1) test statistic matrix
          'results'   — raw Tigramite results object
    """
    N = data.shape[1]
    if var_names is None:
        var_names = [f"X{i}" for i in range(N)]

    dataframe = pp.DataFrame(data, var_names=var_names)
    cond_ind_test = _make_cond_ind_test(test)

    if method == "pcmci":
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha, alpha_level=alpha)
        # Tigramite returns p_matrix[i, j, tau] for the edge X_i(t-tau) -> X_j(t).
        links = extract_links(results, N, alpha)
    elif method == "pcmci_plus":
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
        links = links_from_graph(results["graph"])
    elif method == "lpcmci":
        pcmci = LPCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
        results = pcmci.run_lpcmci(tau_max=tau_max, pc_alpha=pc_alpha)
        links = links_from_graph(results["graph"])
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose 'pcmci', 'pcmci_plus', or 'lpcmci'.")

    return {
        "links": links,
        "p_matrix": results["p_matrix"],
        "val_matrix": results["val_matrix"],
        "q_matrix": results.get("q_matrix"),
        "results": results,
        "pcmci_obj": pcmci,
        "var_names": var_names,
        "method": method,
    }


def extract_links(results: dict, N: int, alpha: float) -> dict:
    """
    Convert Tigramite results to {j: [(i, lag), ...]} format.

    lag > 0 throughout (positive integer), meaning X_i at time t-lag causes X_j at t.
    """
    p_matrix = results["p_matrix"]  # shape (N, N, tau_max+1)
    tau_max = p_matrix.shape[2] - 1

    links = {j: [] for j in range(N)}
    for j in range(N):
        for i in range(N):
            for tau in range(1, tau_max + 1):  # skip tau=0 (contemporaneous)
                if p_matrix[i, j, tau] <= alpha:
                    links[j].append((i, tau))
    return links


def links_from_matrix(
    A: np.ndarray,
    threshold: float = 1e-8,
    lags: int = 1,
) -> dict:
    """
    Extract causal links from a known VAR coefficient matrix.

    Args:
        A: (N, N) matrix where A[i, j] != 0 means X_i at lag `lags` causes X_j
        threshold: minimum absolute value to consider non-zero
        lags: which lag this matrix corresponds to

    Returns:
        {j: [(i, lags), ...]}
    """
    N = A.shape[0]
    links = {j: [] for j in range(N)}
    for j in range(N):
        for i in range(N):
            if abs(A[i, j]) > threshold:
                links[j].append((i, lags))
    return links


def count_links(links: dict) -> int:
    return sum(len(v) for v in links.values())


def compare_links(true_links: dict, estimated_links: dict, N: int, tau_max: int,
                  lag_agnostic: bool = False) -> dict:
    """
    Compute precision, recall, and F1 between true and estimated link sets.

    lag_agnostic: if True, collapse each edge to a directed pair (i, j) and ignore
    the lag. Use this when only the *direction* of a link is ground truth and the
    exact lag is not (e.g. river travel time): an edge found at the right direction
    but a different lag then counts as a hit instead of a miss.
    """
    if lag_agnostic:
        true_set = {(i, j) for j, lst in true_links.items() for (i, _) in lst}
        est_set  = {(i, j) for j, lst in estimated_links.items() for (i, _) in lst}
    else:
        true_set = {(i, j, tau) for j, lst in true_links.items() for (i, tau) in lst}
        est_set  = {(i, j, tau) for j, lst in estimated_links.items() for (i, tau) in lst}

    tp = len(true_set & est_set)
    fp = len(est_set - true_set)
    fn = len(true_set - est_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
