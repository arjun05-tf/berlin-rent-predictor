"""Tests for preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from berlinrentml.features.preprocessing import LeakageSafePreprocessor, create_preprocessor


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = LeakageSafePreprocessor(
        numeric_features=["a", "b"],
        categorical_features=["c"],
    )

    assert preprocessor.numeric_features == ["a", "b"]
    assert preprocessor.categorical_features == ["c"]
    assert not preprocessor._is_fitted


def test_preprocessor_fit_transform():
    """Test fitting and transforming."""
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "num2": [10, 20, 30, 40],
        "cat1": ["a", "b", "a", "c"],
    })

    preprocessor = LeakageSafePreprocessor(
        numeric_features=["num1", "num2"],
        categorical_features=["cat1"],
    )

    X_transformed = preprocessor.fit_transform(df)

    assert preprocessor._is_fitted
    assert X_transformed.shape[0] == 4
    assert X_transformed.shape[1] > 2  # Scaled numerics + one-hot encoded categoricals


def test_preprocessor_handles_missing():
    """Test that preprocessor handles missing values."""
    df_train = pd.DataFrame({
        "num1": [1, 2, np.nan, 4],
        "cat1": ["a", "b", None, "c"],
    })

    df_test = pd.DataFrame({
        "num1": [5, np.nan],
        "cat1": ["a", None],
    })

    preprocessor = LeakageSafePreprocessor(
        numeric_features=["num1"],
        categorical_features=["cat1"],
    )

    # Fit on training data
    X_train = preprocessor.fit_transform(df_train)
    assert not np.isnan(X_train).any()

    # Transform test data
    X_test = preprocessor.transform(df_test)
    assert not np.isnan(X_test).any()


def test_preprocessor_handles_unknown_categories():
    """Test handling of unknown categories in test set."""
    df_train = pd.DataFrame({
        "cat1": ["a", "b", "a", "b"],
    })

    df_test = pd.DataFrame({
        "cat1": ["a", "c"],  # 'c' is unknown
    })

    preprocessor = LeakageSafePreprocessor(
        numeric_features=[],
        categorical_features=["cat1"],
        handle_unknown="ignore",
    )

    X_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)

    # Should not raise error and should handle unknown category
    assert X_test.shape[0] == 2


def test_transform_before_fit_raises_error():
    """Test that transform before fit raises error."""
    df = pd.DataFrame({"num1": [1, 2, 3]})

    preprocessor = LeakageSafePreprocessor(
        numeric_features=["num1"],
        categorical_features=[],
    )

    with pytest.raises(RuntimeError, match="must be fitted"):
        preprocessor.transform(df)


def test_create_preprocessor():
    """Test preprocessor creation helper."""
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "num2": [10.0, 20.0, 30.0],
        "cat1": ["a", "b", "c"],
    })

    preprocessor = create_preprocessor(df)

    # Should auto-detect numeric and categorical features
    assert "num1" in preprocessor.numeric_features
    assert "num2" in preprocessor.numeric_features
    assert "cat1" in preprocessor.categorical_features


def test_preprocessing_is_fitted_on_training_only():
    """
    Critical test: Ensure preprocessing is fitted only on training data.
    This prevents leakage.
    """
    # Training data: numbers 1-4
    df_train = pd.DataFrame({"num1": [1, 2, 3, 4]})

    # Test data: numbers 10-11 (different distribution)
    df_test = pd.DataFrame({"num1": [10, 11]})

    preprocessor = LeakageSafePreprocessor(
        numeric_features=["num1"],
        categorical_features=[],
    )

    # Fit on training data
    X_train = preprocessor.fit_transform(df_train)

    # Transform test data with training statistics
    X_test = preprocessor.transform(df_test)

    # The test data should be scaled using training mean/std
    # So test values should be different from what they'd be if fitted on test data
    assert X_test.mean() != 0  # Would be 0 if fitted on test data
