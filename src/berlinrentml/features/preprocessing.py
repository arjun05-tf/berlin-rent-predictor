"""Preprocessing pipeline that prevents leakage."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LeakageSafePreprocessor:
    """
    Preprocessing pipeline that prevents target leakage.

    All transformations are fitted on training data only.
    """

    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        handle_unknown: str = "ignore",
    ):
        """
        Initialize preprocessor.

        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            handle_unknown: How to handle unknown categories ('ignore' or 'error')
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.handle_unknown = handle_unknown
        self.preprocessor: Optional[ColumnTransformer] = None
        self._is_fitted = False

    def build_pipeline(self) -> ColumnTransformer:
        """Build preprocessing pipeline."""
        # Numeric pipeline: impute then scale
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Categorical pipeline: impute then one-hot encode
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown=self.handle_unknown,
                        sparse_output=False,
                        drop="first",  # Drop first category to avoid multicollinearity
                    ),
                ),
            ]
        )

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ],
            remainder="drop",  # Drop any other columns
        )

        return preprocessor

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit preprocessor on training data.

        CRITICAL: This must only be called on training data to prevent leakage.

        Args:
            X: Training features
            y: Training target (unused, for sklearn compatibility)
        """
        self.preprocessor = self.build_pipeline()
        self.preprocessor.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.

        Args:
            X: Features to transform

        Returns:
            Transformed feature array
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit preprocessor and transform training data.

        Args:
            X: Training features
            y: Training target (unused, for sklearn compatibility)

        Returns:
            Transformed feature array
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before getting feature names")

        return self.preprocessor.get_feature_names_out().tolist()


def create_preprocessor(
    X: pd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> LeakageSafePreprocessor:
    """
    Create preprocessor from DataFrame.

    Args:
        X: Input DataFrame
        numeric_features: List of numeric features (auto-detected if None)
        categorical_features: List of categorical features (auto-detected if None)

    Returns:
        LeakageSafePreprocessor instance
    """
    if numeric_features is None:
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_features is None:
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Ensure features exist in DataFrame
    numeric_features = [f for f in numeric_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]

    return LeakageSafePreprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        handle_unknown="ignore",
    )
