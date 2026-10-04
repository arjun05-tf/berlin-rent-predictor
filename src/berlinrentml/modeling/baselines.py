"""Baseline models."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class MeanPredictor(BaseEstimator, RegressorMixin):
    """Predict the mean of training data."""

    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        """Fit by computing mean of y."""
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        """Predict mean for all samples."""
        return np.full(len(X), self.mean_)


class MedianPredictor(BaseEstimator, RegressorMixin):
    """Predict the median of training data."""

    def __init__(self):
        self.median_ = None

    def fit(self, X, y):
        """Fit by computing median of y."""
        self.median_ = np.median(y)
        return self

    def predict(self, X):
        """Predict median for all samples."""
        return np.full(len(X), self.median_)
