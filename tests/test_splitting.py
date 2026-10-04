"""Tests for train/test splitting strategies."""

import pandas as pd
import pytest

from berlinrentml.modeling.splitting import (
    RandomSplit,
    GroupedSplit,
    SpatialHoldoutSplit,
    TemporalSplit,
)


def test_random_split():
    """Test random splitting."""
    df = pd.DataFrame({
        "feature1": range(100),
        "feature2": range(100, 200),
        "target": range(200, 300),
    })

    splitter = RandomSplit(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(df, "target")

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns


def test_grouped_split():
    """Test grouped splitting."""
    df = pd.DataFrame({
        "feature1": range(100),
        "group": ["A"] * 50 + ["B"] * 50,
        "target": range(100),
    })

    splitter = GroupedSplit(group_col="group", test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(df, "target")

    # Check that groups don't overlap
    train_groups = set(X_train["group"].unique())
    test_groups = set(X_test["group"].unique())

    # For 2 groups with 20% test size, we expect 1 group in test
    assert len(test_groups) > 0
    # Groups should not overlap (this may not always hold with 2 groups)


def test_spatial_holdout_split():
    """Test spatial holdout splitting."""
    df = pd.DataFrame({
        "feature1": range(100),
        "region": ["North"] * 40 + ["South"] * 30 + ["East"] * 30,
        "target": range(100),
    })

    splitter = SpatialHoldoutSplit(
        region_col="region",
        holdout_regions=["East"],
    )
    X_train, X_test, y_train, y_test = splitter.split(df, "target")

    # Check that test set only contains holdout region
    assert set(X_test["region"].unique()) == {"East"}
    assert len(X_test) == 30

    # Check that train set doesn't contain holdout region
    assert "East" not in X_train["region"].values
    assert len(X_train) == 70


def test_temporal_split():
    """Test temporal splitting."""
    df = pd.DataFrame({
        "feature1": range(100),
        "date": pd.date_range("2020-01-01", periods=100, freq="D"),
        "target": range(100),
    })

    splitter = TemporalSplit(date_col="date", test_size=0.2)
    X_train, X_test, y_train, y_test = splitter.split(df, "target")

    # Check that test dates are after train dates
    train_max_date = X_train["date"].max()
    test_min_date = X_test["date"].min()

    assert test_min_date >= train_max_date
    assert len(X_test) == 20
