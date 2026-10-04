"""Feature engineering utilities."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class FeatureEngineer:
    """Engineer features for rental price prediction."""

    def __init__(self):
        self.feature_names: List[str] = []
        self.encoders: dict = {}
        self.scalers: dict = {}

    def create_age_features(self, df: pd.DataFrame, year_col: str = "yearConstructed") -> pd.DataFrame:
        """
        Create building age features.

        Args:
            df: DataFrame
            year_col: Column with construction year

        Returns:
            DataFrame with age features
        """
        df = df.copy()

        if year_col in df.columns:
            current_year = 2026  # Use fixed year for reproducibility
            df["building_age"] = current_year - df[year_col]

            # Clip age to reasonable range
            df["building_age"] = df["building_age"].clip(lower=0, upper=200)

            # Create age bins
            df["age_bin"] = pd.cut(
                df["building_age"],
                bins=[0, 5, 20, 50, 100, 200],
                labels=["very_new", "new", "medium", "old", "very_old"],
            )

        return df

    def create_size_features(self, df: pd.DataFrame, size_col: str = "livingSpace") -> pd.DataFrame:
        """
        Create size-related features.

        Args:
            df: DataFrame
            size_col: Column with living space

        Returns:
            DataFrame with size features
        """
        df = df.copy()

        if size_col in df.columns:
            # Size bins
            df["size_bin"] = pd.cut(
                df[size_col],
                bins=[0, 40, 70, 100, 500],
                labels=["small", "medium", "large", "very_large"],
            )

            # Log size (often linear relationship with log price)
            df["log_size"] = np.log1p(df[size_col])

        # Rooms per m² (space efficiency)
        if "rooms" in df.columns and size_col in df.columns:
            df["rooms_per_m2"] = df["rooms"] / df[size_col]
            df["rooms_per_m2"] = df["rooms_per_m2"].clip(upper=0.1)  # Cap extreme values

        return df

    def create_location_features(
        self, df: pd.DataFrame, district_col: str = "geo_bln", plz_col: str = "geo_plz"
    ) -> pd.DataFrame:
        """
        Create location features.

        IMPORTANT: These must be fitted on training data only to avoid leakage.

        Args:
            df: DataFrame
            district_col: District column
            plz_col: Postal code column

        Returns:
            DataFrame with location features
        """
        df = df.copy()

        # For now, just pass through - encoding will happen in preprocessing
        # We explicitly do NOT create target-based location features here
        # (like mean rent by district) as that would leak information

        return df

    def create_amenity_score(
        self,
        df: pd.DataFrame,
        amenity_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create amenity score by counting available amenities.

        Args:
            df: DataFrame
            amenity_cols: List of boolean amenity columns

        Returns:
            DataFrame with amenity_score
        """
        df = df.copy()

        if amenity_cols is None:
            amenity_cols = ["hasKitchen", "hasBalcony", "hasGarden", "cellar"]

        # Only use columns that exist
        existing_amenity_cols = [col for col in amenity_cols if col in df.columns]

        if existing_amenity_cols:
            # Convert to numeric (True=1, False=0) and sum
            amenity_df = df[existing_amenity_cols].fillna(False).astype(int)
            df["amenity_score"] = amenity_df.sum(axis=1)

        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        create_age: bool = True,
        create_size: bool = True,
        create_location: bool = True,
        create_amenities: bool = True,
    ) -> pd.DataFrame:
        """
        Apply all feature engineering.

        Args:
            df: Input DataFrame
            create_age: Create age features
            create_size: Create size features
            create_location: Create location features
            create_amenities: Create amenity features

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        if create_age:
            df = self.create_age_features(df)

        if create_size:
            df = self.create_size_features(df)

        if create_location:
            df = self.create_location_features(df)

        if create_amenities:
            df = self.create_amenity_score(df)

        return df


def select_features(
    df: pd.DataFrame,
    feature_config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features for modeling.

    Args:
        df: DataFrame with engineered features
        feature_config: Configuration dict with feature lists

    Returns:
        (selected_features_df, feature_names)
    """
    if feature_config is None:
        # Default feature configuration
        feature_config = {
            "numeric": [
                "livingSpace",
                "rooms",
                "floor",
                "building_age",
                "log_size",
                "rooms_per_m2",
                "amenity_score",
            ],
            "categorical": [
                "geo_plz",
                "geo_bln",
                "heatingType",
                "condition",
                "age_bin",
                "size_bin",
            ],
            "boolean": [
                "hasKitchen",
                "hasBalcony",
                "hasGarden",
                "cellar",
            ],
        }

    all_features = []
    for feature_list in feature_config.values():
        all_features.extend(feature_list)

    # Only keep features that exist in the DataFrame
    available_features = [f for f in all_features if f in df.columns]

    return df[available_features].copy(), available_features
