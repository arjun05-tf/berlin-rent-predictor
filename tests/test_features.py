"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from berlinrentml.features import FeatureEngineer, select_features


def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization."""
    engineer = FeatureEngineer()
    assert engineer.feature_names == []
    assert engineer.encoders == {}


def test_create_age_features():
    """Test building age features."""
    df = pd.DataFrame({
        "yearConstructed": [2000, 1990, 1950, 2020],
    })

    engineer = FeatureEngineer()
    df_with_age = engineer.create_age_features(df)

    assert "building_age" in df_with_age.columns
    assert df_with_age["building_age"].iloc[0] == 26  # 2026 - 2000
    assert "age_bin" in df_with_age.columns


def test_create_size_features():
    """Test size features."""
    df = pd.DataFrame({
        "livingSpace": [30, 50, 80, 120],
        "rooms": [1, 2, 3, 5],
    })

    engineer = FeatureEngineer()
    df_with_size = engineer.create_size_features(df)

    assert "size_bin" in df_with_size.columns
    assert "log_size" in df_with_size.columns
    assert "rooms_per_m2" in df_with_size.columns

    # Check log transformation
    assert df_with_size["log_size"].iloc[0] == np.log1p(30)


def test_create_amenity_score():
    """Test amenity score creation."""
    df = pd.DataFrame({
        "hasKitchen": [True, True, False, True],
        "hasBalcony": [True, False, True, True],
        "hasGarden": [False, False, False, True],
        "cellar": [True, True, False, False],
    })

    engineer = FeatureEngineer()
    df_with_amenity = engineer.create_amenity_score(df)

    assert "amenity_score" in df_with_amenity.columns
    assert df_with_amenity["amenity_score"].iloc[0] == 3  # Kitchen, balcony, cellar
    assert df_with_amenity["amenity_score"].iloc[3] == 3  # Kitchen, balcony, garden


def test_engineer_features_full():
    """Test full feature engineering pipeline."""
    df = pd.DataFrame({
        "livingSpace": [50, 80],
        "rooms": [2, 3],
        "yearConstructed": [2000, 1990],
        "hasKitchen": [True, True],
        "hasBalcony": [True, False],
    })

    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df)

    # Check that all feature types are created
    assert "building_age" in df_engineered.columns
    assert "size_bin" in df_engineered.columns
    assert "amenity_score" in df_engineered.columns


def test_select_features():
    """Test feature selection."""
    df = pd.DataFrame({
        "livingSpace": [50, 80, 60],
        "rooms": [2, 3, 2],
        "building_age": [26, 36, 10],
        "geo_plz": ["10115", "10117", "10119"],
        "extra_column": [1, 2, 3],
    })

    feature_config = {
        "numeric": ["livingSpace", "rooms", "building_age"],
        "categorical": ["geo_plz"],
    }

    df_selected, feature_names = select_features(df, feature_config)

    assert len(df_selected.columns) == 4
    assert "extra_column" not in df_selected.columns
    assert "livingSpace" in feature_names


def test_select_features_missing_columns():
    """Test feature selection with missing columns."""
    df = pd.DataFrame({
        "livingSpace": [50, 80, 60],
        "rooms": [2, 3, 2],
    })

    feature_config = {
        "numeric": ["livingSpace", "rooms", "nonexistent_feature"],
    }

    df_selected, feature_names = select_features(df, feature_config)

    # Should only include existing features
    assert len(feature_names) == 2
    assert "nonexistent_feature" not in feature_names
