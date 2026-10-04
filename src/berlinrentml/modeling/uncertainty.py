"""Split-conformal prediction intervals in log-rent space."""

import numpy as np


class ConformalInterval:
    """Multiplicative interval around a point prediction with finite-sample coverage.

    Fit on a calibration set whose groups (postal codes) were unseen by the model,
    so coverage holds for new neighbourhoods, not just new rows of known ones.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_: float | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalInterval":
        scores = np.abs(np.log1p(y_true) - np.log1p(y_pred))
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(scores, level))
        return self

    def predict(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_ is None:
            raise RuntimeError("Call fit first")
        z = np.log1p(y_pred)
        return np.expm1(z - self.q_), np.expm1(z + self.q_)

    def coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        lo, hi = self.predict(y_pred)
        return float(np.mean((y_true >= lo) & (y_true <= hi)))
