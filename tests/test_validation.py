"""Unit tests for data validation."""

import numpy as np
import pandas as pd
import pytest

from berlinrentml.data.validation import DataValidator, validate_data


def test_validator_initialization():
    """Test DataValidator initialization."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = DataValidator(df)
    assert len(validator.df) == 3
    assert len(validator.issues) == 0


def test_validate_schema():
    """Test schema validation."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    validator = DataValidator(df)

    # Test with all required columns present
    is_valid, missing = validator.validate_schema(["a", "b"])
    assert is_valid
    assert len(missing) == 0

    # Test with missing column
    is_valid, missing = validator.validate_schema(["a", "b", "c"])
    assert not is_valid
    assert "c" in missing


def test_check_missing_values():
    """Test missing value detection."""
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, None, None, None, 5],
    })
    validator = DataValidator(df)

    # Check with 50% threshold
    problematic = validator.check_missing_values(threshold=0.5)

    # Column b has 80% missing, should be flagged
    assert "b" in problematic
    assert problematic["b"] == 0.8

    # Column a has 20% missing, should not be flagged
    assert "a" not in problematic


def test_check_duplicates():
    """Test duplicate detection."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 1],
        "b": [4, 5, 6, 4],
    })
    validator = DataValidator(df)

    n_duplicates = validator.check_duplicates()
    assert n_duplicates == 1


def test_check_value_ranges():
    """Test value range validation."""
    df = pd.DataFrame({"price": [100, 200, 300, 5000, 10]})
    validator = DataValidator(df)

    # Check for values outside reasonable range
    n_issues = validator.check_value_ranges("price", min_val=50, max_val=4000)
    assert n_issues == 2  # One below 50, one above 4000


def test_check_target_variable():
    """Test target variable validation."""
    df = pd.DataFrame({"rent": [500, 800, 1200, -100, 0, 2000]})
    validator = DataValidator(df)

    stats = validator.check_target_variable("rent")

    assert stats["count"] == 6
    assert stats["n_negative"] == 1
    assert stats["n_zero"] == 1
    assert stats["min"] == -100
    assert stats["max"] == 2000


def test_validate_data():
    """Test full data validation."""
    df = pd.DataFrame({
        "baseRent": [500, 800, 1200, 1500],
        "livingSpace": [40, 60, 80, 100],
        "rooms": [2, 3, 4, 5],
    })

    is_valid, summary = validate_data(df, target_col="baseRent")
    assert is_valid
    assert "No validation issues" in summary


def test_validate_data_with_issues():
    """Test validation with issues."""
    df = pd.DataFrame({
        "baseRent": [500, -100, 1200, None],
        "livingSpace": [40, 60, None, 100],
    })

    is_valid, summary = validate_data(
        df,
        target_col="baseRent",
        required_columns=["baseRent", "livingSpace"],
    )
    assert not is_valid
    assert "validation issues" in summary.lower()
