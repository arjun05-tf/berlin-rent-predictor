"""Train/test splitting strategies with geographic awareness."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit


class SplitStrategy:
    """Base class for splitting strategies."""

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test.

        Returns:
            X_train, X_test, y_train, y_test
        """
        raise NotImplementedError


class RandomSplit(SplitStrategy):
    """Random train/test split."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test


class GroupedSplit(SplitStrategy):
    """Grouped train/test split by location."""

    def __init__(
        self,
        group_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize grouped split.

        Args:
            group_col: Column to group by (e.g., 'geo_plz', 'geo_bln')
            test_size: Proportion of groups in test set
            random_state: Random seed
        """
        self.group_col = group_col
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.group_col not in df.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in DataFrame")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        groups = df[self.group_col]

        # Use GroupShuffleSplit
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )

        train_idx, test_idx = next(splitter.split(X, y, groups))

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        return X_train, X_test, y_train, y_test


class SpatialHoldoutSplit(SplitStrategy):
    """Hold out specific geographic regions for testing."""

    def __init__(
        self,
        region_col: str,
        holdout_regions: List[str],
    ):
        """
        Initialize spatial holdout split.

        Args:
            region_col: Column containing region identifiers
            holdout_regions: List of regions to hold out for testing
        """
        self.region_col = region_col
        self.holdout_regions = holdout_regions

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.region_col not in df.columns:
            raise ValueError(f"Region column '{self.region_col}' not found in DataFrame")

        # Create boolean mask for test regions
        test_mask = df[self.region_col].isin(self.holdout_regions)

        train_df = df[~test_mask]
        test_df = df[test_mask]

        if len(test_df) == 0:
            raise ValueError(
                f"No samples found in holdout regions: {self.holdout_regions}. "
                f"Available regions: {df[self.region_col].unique().tolist()}"
            )

        X_train = train_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        print(f"Spatial holdout: Training on {len(train_df):,} samples")
        print(f"Spatial holdout: Testing on {len(test_df):,} samples in regions: {self.holdout_regions}")

        return X_train, X_test, y_train, y_test


class TemporalSplit(SplitStrategy):
    """Temporal train/test split."""

    def __init__(
        self,
        date_col: str,
        test_start_date: Optional[str] = None,
        test_size: float = 0.2,
    ):
        """
        Initialize temporal split.

        Args:
            date_col: Column containing dates
            test_start_date: Date to start test set (YYYY-MM-DD). If None, uses test_size.
            test_size: If test_start_date is None, proportion of most recent data for test
        """
        self.date_col = date_col
        self.test_start_date = test_start_date
        self.test_size = test_size

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.date_col not in df.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in DataFrame")

        # Ensure date column is datetime
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        if self.test_start_date is not None:
            # Split by specific date
            test_date = pd.to_datetime(self.test_start_date)
            test_mask = df[self.date_col] >= test_date
        else:
            # Split by quantile
            date_threshold = df[self.date_col].quantile(1 - self.test_size)
            test_mask = df[self.date_col] >= date_threshold

        train_df = df[~test_mask]
        test_df = df[test_mask]

        if len(test_df) == 0:
            raise ValueError("No samples in test set. Check date column and split parameters.")

        X_train = train_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        print(f"Temporal split: Training on {len(train_df):,} samples (before {test_date if self.test_start_date else date_threshold})")
        print(f"Temporal split: Testing on {len(test_df):,} samples (from {test_date if self.test_start_date else date_threshold})")

        return X_train, X_test, y_train, y_test


def get_splitter(
    strategy: str,
    **kwargs,
) -> SplitStrategy:
    """
    Get splitting strategy by name.

    Args:
        strategy: One of 'random', 'grouped', 'spatial', 'temporal'
        **kwargs: Arguments for the splitter

    Returns:
        SplitStrategy instance
    """
    if strategy == "random":
        return RandomSplit(**kwargs)
    elif strategy == "grouped":
        return GroupedSplit(**kwargs)
    elif strategy == "spatial":
        return SpatialHoldoutSplit(**kwargs)
    elif strategy == "temporal":
        return TemporalSplit(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
