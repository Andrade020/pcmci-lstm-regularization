"""
Causal discovery utilities wrapping Tigramite's PCMCI.
"""
import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr


def run_pcmci(
    data: np.ndarray,
    tau_max: int = 5,
    alpha: float = 0.05,
    pc_alpha: float = 0.1,
    test: str = "parcorr",
    var_names: list = None,
) -> dict:
    """
    Run PCMCI on a multivariate time series and return the significant causal links.

    Args:
        data: (T, N) normalized numpy array (should be stationary)
        tau_max: maximum lag to test
        alpha: significance threshold for final MCI test
        pc_alpha: significance threshold for PC1 phase (liberal)
        test: 'parcorr' or 'robust_parcorr'
        var_names: optional list of variable names for output

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

    if test == "parcorr":
        cond_ind_test = ParCorr()
    elif test == "robust_parcorr":
        cond_ind_test = RobustParCorr()
    else:
        raise ValueError(f"Unknown test: {test}. Choose 'parcorr' or 'robust_parcorr'.")

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha, alpha_level=alpha)

    # Extract significant links as dict {j: [(i, lag), ...]}
    # Note: Tigramite returns links[j] as list of tuples (i, -tau) with tau > 0
    links = extract_links(results, N, alpha)

    return {
        "links": links,
        "p_matrix": results["p_matrix"],
        "val_matrix": results["val_matrix"],
        "q_matrix": results.get("q_matrix"),
        "results": results,
        "pcmci_obj": pcmci,
        "var_names": var_names,
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


def compare_links(true_links: dict, estimated_links: dict, N: int, tau_max: int) -> dict:
    """
    Compute precision, recall, and F1 between true and estimated link sets.
    """
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
