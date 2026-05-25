"""
Evaluation metrics, statistical tests, and ablation utilities.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MSE, RMSE, and MAE for multivariate predictions.

    Args:
        y_true, y_pred: (T, N) arrays (already in original scale)

    Returns:
        dict with 'mse', 'rmse', 'mae' (averaged over all variables)
    """
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"mse": mse, "rmse": rmse, "mae": mae}


def compute_metrics_per_var(y_true: np.ndarray, y_pred: np.ndarray, var_names: list = None) -> dict:
    """Return per-variable metrics."""
    N = y_true.shape[1]
    if var_names is None:
        var_names = [f"X{i}" for i in range(N)]
    out = {}
    for i, name in enumerate(var_names):
        out[name] = compute_metrics(y_true[:, i:i+1], y_pred[:, i:i+1])
    return out


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (averaged over all elements)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))) * 100


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
) -> tuple:
    """
    Harvey, Leybourne & Newbold (1997) modified Diebold-Mariano test.

    Tests H0: equal predictive accuracy between two forecast sequences.
    Positive DM statistic means e1 has higher squared loss (model 1 is worse).
    Negative DM statistic means model 1 is better.

    Args:
        e1, e2: forecast error arrays, shape (T,) or (T, N).
                If (T, N), they are flattened (pooled over variables).
        h: forecast horizon (1 for one-step-ahead).

    Returns:
        (dm_stat, p_value) — two-sided p-value under t_{T-1}.
    """
    if e1.ndim > 1:
        e1 = e1.ravel()
        e2 = e2.ravel()

    d = e1 ** 2 - e2 ** 2          # loss differential (squared errors)
    T = len(d)
    d_bar = d.mean()

    # Long-run variance via Newey-West with (h-1) lags
    gamma_0 = np.sum((d - d_bar) ** 2) / T
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / T
        gamma_sum += gamma_k
    lrv = (gamma_0 + 2.0 * gamma_sum) / T   # variance of d_bar

    if lrv <= 0:
        return float("nan"), float("nan")

    dm = d_bar / np.sqrt(lrv)

    # HLN small-sample correction factor
    hlm_factor = np.sqrt((T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T)
    dm_hlm = dm * hlm_factor

    p_value = float(2.0 * stats.t.sf(abs(dm_hlm), df=T - 1))
    return float(dm_hlm), p_value


def block_bootstrap_ci(
    losses: np.ndarray,
    block_size: int = 12,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple:
    """
    Block bootstrap confidence interval for the mean loss.

    Accounts for temporal dependence by sampling contiguous blocks.

    Args:
        losses: (T,) squared error array.
        block_size: length of each bootstrap block (12 = 1 year for monthly data).
        n_bootstrap: number of bootstrap replications.
        ci: confidence level (e.g. 0.95 for 95% CI).
        seed: random seed.

    Returns:
        (lower, upper) confidence bounds on the mean loss.
    """
    rng = np.random.default_rng(seed)
    T = len(losses)
    alpha = (1.0 - ci) / 2.0

    # Possible block start positions
    starts = np.arange(0, T - block_size + 1)
    n_blocks_needed = int(np.ceil(T / block_size))

    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(starts, size=n_blocks_needed, replace=True)
        sample = np.concatenate([losses[s: s + block_size] for s in idx])[:T]
        boot_means[b] = sample.mean()

    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return lower, upper
