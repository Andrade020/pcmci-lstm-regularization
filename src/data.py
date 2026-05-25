"""
Data loading, preprocessing, and dataset utilities.
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch


# ── Quarterly series (legacy, kept for reproducibility) ───────────────────────
FRED_SERIES = {
    "GDPC1":    "Real GDP",
    "PCECC96":  "Real PCE",
    "GPDIC1":   "Real Investment",
    "CPIAUCSL": "CPI",
    "UNRATE":   "Unemployment Rate",
    "FEDFUNDS": "Fed Funds Rate",
    "INDPRO":   "Industrial Production",
    "M2SL":     "M2 Money Supply",
    "GS10":     "10yr Treasury Yield",
    "HOUST":    "Housing Starts",
}

# ── Monthly series (main experiment) ──────────────────────────────────────────
# Based on McCracken & Ng (2016) FRED-MD subset.
# Value = (label, transform) where transform is 'log_diff' or 'diff'.
FRED_MONTHLY = {
    "INDPRO":   ("Industrial Production",  "log_diff"),
    "PAYEMS":   ("Nonfarm Payrolls",        "log_diff"),
    "UNRATE":   ("Unemployment Rate",       "diff"),
    "HOUST":    ("Housing Starts",          "log_diff"),
    "CPIAUCSL": ("CPI",                     "log_diff"),
    "PPIACO":   ("PPI All Commodities",     "log_diff"),
    "FEDFUNDS": ("Fed Funds Rate",          "diff"),
    "GS10":     ("10yr Treasury",           "diff"),
    "GS1":      ("1yr Treasury",            "diff"),
    "M2SL":     ("M2 Money Supply",         "log_diff"),
    "TOTALSL":  ("Consumer Credit",         "log_diff"),
    "BUSLOANS": ("Business Loans",          "log_diff"),
}


def load_fred_monthly(
    series: dict = None,
    start: str = "1960-01-01",
    end: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download and preprocess the monthly FRED macro panel (McCracken-Ng subset).

    Each series in `series` has its own transformation code ('log_diff' or 'diff')
    to achieve stationarity, following standard FRED-MD practice.

    Args:
        series: dict {ticker: (label, transform)}, defaults to FRED_MONTHLY.
        start: start date.
        end: end date (defaults to today).

    Returns:
        (raw_df, transformed_df) indexed by date, monthly frequency.
    """
    if series is None:
        series = FRED_MONTHLY
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    tickers = list(series.keys())
    raw = pdr.DataReader(tickers, "fred", start, end)

    # Resample to month-start, keep last observation of each month
    raw = raw.resample("MS").last()

    # Rename columns
    raw.columns = [series[t][0] for t in tickers]

    # Apply per-column transformations
    transformed_cols = {}
    for t in tickers:
        label, tf = series[t]
        col = raw[label]
        if tf == "log_diff":
            transformed_cols[label] = np.log(col).diff()
        elif tf == "diff":
            transformed_cols[label] = col.diff()
        else:
            raise ValueError(f"Unknown transform '{tf}' for {t}")

    transformed = pd.DataFrame(transformed_cols, index=raw.index).dropna()
    raw = raw.loc[transformed.index]   # align raw to same index
    return raw, transformed


def load_fred_data(
    series: dict = None,
    start: str = "1960-01-01",
    end: str = None,
    freq: str = "QS",
    transform: str = "log_diff",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download and preprocess FRED macroeconomic series.

    Args:
        series: dict {ticker: label}. Defaults to FRED_SERIES.
        start: start date string
        end: end date string (defaults to today)
        freq: resample frequency
        transform: 'log_diff' (growth rates) or 'diff' (first differences)

    Returns:
        (raw_df, transformed_df) — both indexed by date
    """
    if series is None:
        series = FRED_SERIES
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    tickers = list(series.keys())
    raw = pdr.DataReader(tickers, "fred", start, end).dropna()

    # resample to quarterly if needed
    if raw.index.freq is None or str(raw.index.freq) != freq:
        raw = raw.resample(freq).last().dropna()

    raw.columns = [series[t] for t in tickers if t in series]

    if transform == "log_diff":
        transformed = np.log(raw).diff().dropna()
    elif transform == "diff":
        transformed = raw.diff().dropna()
    else:
        raise ValueError(f"Unknown transform: {transform}")

    return raw, transformed


def generate_var_series(
    A: np.ndarray,
    n_steps: int,
    noise_std: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a multivariate time series from a VAR(1) model:
        x_t = A @ x_{t-1} + ε_t,  ε_t ~ N(0, noise_std² I)

    Args:
        A: (N, N) coefficient matrix. Spectral radius should be < 1 for stationarity.
        n_steps: total length of the series
        noise_std: standard deviation of innovations
        seed: random seed

    Returns:
        array of shape (n_steps, N)
    """
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    X = np.zeros((n_steps, N))
    for t in range(1, n_steps):
        X[t] = A @ X[t - 1] + rng.normal(scale=noise_std, size=N)
    return X


def train_test_split_ts(
    data: np.ndarray,
    train_frac: float = 0.8,
    window_len: int = 5,
) -> tuple:
    """
    Split a time series array into train and test sets, keeping the last
    `window_len` steps of training in the test array (for the initial window).

    Returns:
        train_raw, test_raw — numpy arrays
    """
    n_train = int(len(data) * train_frac)
    return data[:n_train], data[n_train - window_len:]


class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for one-step-ahead multivariate forecasting.

    Args:
        data: (T, N) normalized numpy array
        window_len: input sequence length L
    """

    def __init__(self, data: np.ndarray, window_len: int):
        self.window_len = window_len
        self.X = torch.from_numpy(data).float()

    def __len__(self) -> int:
        return len(self.X) - self.window_len

    def __getitem__(self, idx: int):
        x = self.X[idx: idx + self.window_len]          # (L, N)
        y = self.X[idx + self.window_len]               # (N,)
        return x, y


def fit_scaler(train_raw: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_raw)
    return scaler
