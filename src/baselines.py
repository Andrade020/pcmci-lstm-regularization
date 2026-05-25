"""
Non-neural baseline forecasters.
"""
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR


class RandomWalkBaseline:
    """
    Driftless random walk: predicts the last observed value in the window.

    In log-differenced space this corresponds to predicting zero growth,
    i.e. the level follows a random walk without drift.
    """

    def predict(self, test_norm: np.ndarray, window_len: int) -> np.ndarray:
        """
        Args:
            test_norm: (T_test + window_len, N) normalized array
                       (same layout as TimeSeriesDataset: first `window_len`
                       rows overlap with training end).
            window_len: look-back window used by the LSTM.

        Returns:
            (T_test, N) predictions in normalized scale.
        """
        T_test = len(test_norm) - window_len
        # Predict y_{t+window} = y_{t+window-1} (last value in window)
        preds = test_norm[window_len - 1: window_len - 1 + T_test]
        return preds.copy()


class VARBaseline:
    """
    Vector Autoregression baseline with BIC-selected lag order.

    Fits VAR on normalized training data; makes rolling one-step-ahead
    predictions on the test period without re-fitting.
    """

    def __init__(self, maxlags: int = 6, ic: str = "bic"):
        self.maxlags = maxlags
        self.ic = ic
        self._fitted = None
        self._k_ar = None

    def fit(self, train_norm: np.ndarray):
        model = VAR(train_norm)
        try:
            res = model.fit(maxlags=self.maxlags, ic=self.ic)
        except Exception:
            # Fall back to lag-1 if BIC selection fails (e.g. too few obs)
            res = model.fit(maxlags=1)
        self._fitted = res
        self._k_ar = res.k_ar
        return self

    def predict(self, train_norm: np.ndarray, test_norm: np.ndarray) -> np.ndarray:
        """
        Rolling one-step-ahead forecast using the model fit on train_norm.

        Args:
            train_norm: (T_train, N) normalized training data.
            test_norm: (T_test + window_len, N) — same layout as LSTM test set.
                       Only the T_test targets (after the window overlap) are used.

        Returns:
            (T_test, N) one-step-ahead predictions in normalized scale.
        """
        if self._fitted is None:
            self.fit(train_norm)

        # We only need the actual test targets, not the window overlap
        # test_norm has window_len extra rows at the start from training
        T_test = len(test_norm)  # this already includes window overlap rows

        N = train_norm.shape[1]
        preds = np.zeros((T_test, N))

        # Build history = train_norm + progressively appended test rows
        history = train_norm.copy()
        for t in range(T_test):
            lag = self._k_ar
            fc = self._fitted.forecast(history[-lag:], steps=1)
            preds[t] = fc[0]
            history = np.vstack([history, test_norm[t: t + 1]])

        return preds

    @property
    def lag_order(self) -> int:
        return self._k_ar if self._k_ar is not None else 0
