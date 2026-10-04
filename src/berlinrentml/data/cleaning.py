"""Data cleaning utilities."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataCleaner:
    """Clean and preprocess raw data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_log: List[str] = []

    def handle_missing_values(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
    ) -> pd.DataFrame:
        """
        Handle missing values.

        Args:
            numeric_strategy: Strategy for numeric columns ('median', 'mean', 'drop')
            categorical_strategy: Strategy for categorical columns ('mode', 'unknown', 'drop')

        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(self.df)

        # Identify numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Handle numeric columns
        for col in numeric_cols:
            n_missing = self.df[col].isnull().sum()
            if n_missing > 0:
                if numeric_strategy == "median":
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    self.cleaning_log.append(f"{col}: filled {n_missing} missing with median")
                elif numeric_strategy == "mean":
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    self.cleaning_log.append(f"{col}: filled {n_missing} missing with mean")
                elif numeric_strategy == "drop":
                    self.df.dropna(subset=[col], inplace=True)
                    self.cleaning_log.append(f"{col}: dropped {n_missing} rows with missing")

        # Handle categorical columns
        for col in categorical_cols:
            n_missing = self.df[col].isnull().sum()
            if n_missing > 0:
                if categorical_strategy == "mode":
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val[0], inplace=True)
                        self.cleaning_log.append(f"{col}: filled {n_missing} missing with mode")
                elif categorical_strategy == "unknown":
                    self.df[col].fillna("unknown", inplace=True)
                    self.cleaning_log.append(f"{col}: filled {n_missing} missing with 'unknown'")
                elif categorical_strategy == "drop":
                    self.df.dropna(subset=[col], inplace=True)
                    self.cleaning_log.append(f"{col}: dropped {n_missing} rows with missing")

        final_rows = len(self.df)
        if final_rows < initial_rows:
            self.cleaning_log.append(
                f"Total rows dropped: {initial_rows - final_rows} ({(initial_rows - final_rows)/initial_rows*100:.1f}%)"
            )

        return self.df

    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = "first") -> pd.DataFrame:
        """
        Remove duplicate rows.

        Args:
            subset: Columns to consider for identifying duplicates
            keep: Which duplicates to keep ('first', 'last', False)

        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        final_rows = len(self.df)

        n_removed = initial_rows - final_rows
        if n_removed > 0:
            self.cleaning_log.append(
                f"Removed {n_removed:,} duplicate rows ({n_removed/initial_rows*100:.1f}%)"
            )

        return self.df

    def handle_outliers(
        self,
        column: str,
        method: str = "iqr",
        threshold: float = 3.0,
        action: str = "cap",
    ) -> pd.DataFrame:
        """
        Handle outliers in a numeric column.

        Args:
            column: Column name
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection (IQR multiplier or z-score)
            action: Action to take ('cap', 'remove')

        Returns:
            DataFrame with outliers handled
        """
        if column not in self.df.columns:
            return self.df

        initial_rows = len(self.df)
        series = self.df[column].dropna()

        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        else:
            raise ValueError(f"Unknown method: {method}")

        n_outliers = ((series < lower_bound) | (series > upper_bound)).sum()

        if n_outliers > 0:
            if action == "cap":
                self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
                self.cleaning_log.append(
                    f"{column}: capped {n_outliers} outliers to [{lower_bound:.2f}, {upper_bound:.2f}]"
                )
            elif action == "remove":
                self.df = self.df[
                    (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
                ].copy()
                self.cleaning_log.append(f"{column}: removed {n_outliers} outliers")

        final_rows = len(self.df)
        if final_rows < initial_rows:
            self.cleaning_log.append(f"Rows after outlier handling: {final_rows}")

        return self.df

    def validate_value_ranges(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        action: str = "remove",
    ) -> pd.DataFrame:
        """
        Validate and fix values outside expected range.

        Args:
            column: Column name
            min_val: Minimum valid value
            max_val: Maximum valid value
            action: Action for invalid values ('remove', 'cap', 'nan')

        Returns:
            DataFrame with invalid values handled
        """
        if column not in self.df.columns:
            return self.df

        initial_rows = len(self.df)
        mask = pd.Series([True] * len(self.df), index=self.df.index)

        if min_val is not None:
            mask &= self.df[column] >= min_val

        if max_val is not None:
            mask &= self.df[column] <= max_val

        n_invalid = (~mask).sum()

        if n_invalid > 0:
            if action == "remove":
                self.df = self.df[mask].copy()
                self.cleaning_log.append(f"{column}: removed {n_invalid} invalid values")
            elif action == "cap":
                if min_val is not None:
                    self.df[column] = self.df[column].clip(lower=min_val)
                if max_val is not None:
                    self.df[column] = self.df[column].clip(upper=max_val)
                self.cleaning_log.append(f"{column}: capped {n_invalid} invalid values")
            elif action == "nan":
                self.df.loc[~mask, column] = np.nan
                self.cleaning_log.append(f"{column}: set {n_invalid} invalid values to NaN")

        final_rows = len(self.df)
        if final_rows < initial_rows:
            self.cleaning_log.append(
                f"Rows after range validation: {final_rows} (removed {initial_rows - final_rows})"
            )

        return self.df

    def get_cleaning_summary(self) -> str:
        """Get summary of cleaning operations."""
        if not self.cleaning_log:
            return "No cleaning operations performed"

        return "Cleaning operations:\n" + "\n".join(f"  • {log}" for log in self.cleaning_log)


def clean_berlin_rental_data(
    df: pd.DataFrame,
    target_col: str = "baseRent",
) -> Tuple[pd.DataFrame, str]:
    """
    Clean Berlin rental dataset with domain-specific logic.

    Args:
        df: Raw DataFrame
        target_col: Target column name

    Returns:
        (cleaned_df, cleaning_summary)
    """
    cleaner = DataCleaner(df)

    # Remove rows with missing target
    initial_size = len(cleaner.df)
    cleaner.df = cleaner.df[cleaner.df[target_col].notna()].copy()
    n_removed_target = initial_size - len(cleaner.df)
    if n_removed_target > 0:
        cleaner.cleaning_log.append(
            f"Removed {n_removed_target:,} rows with missing target ({n_removed_target/initial_size*100:.1f}%)"
        )

    # Validate target: must be positive
    cleaner.validate_value_ranges(target_col, min_val=100, max_val=10000, action="remove")

    # Validate livingSpace: must be positive and reasonable
    if "livingSpace" in cleaner.df.columns:
        cleaner.validate_value_ranges("livingSpace", min_val=10, max_val=500, action="remove")

    # Validate rooms: must be positive
    if "rooms" in cleaner.df.columns:
        cleaner.validate_value_ranges("rooms", min_val=0.5, max_val=20, action="remove")

    # Handle outliers in target using IQR
    cleaner.handle_outliers(target_col, method="iqr", threshold=3.0, action="cap")

    # Remove exact duplicates
    cleaner.remove_duplicates()

    summary = cleaner.get_cleaning_summary()
    return cleaner.df, summary
