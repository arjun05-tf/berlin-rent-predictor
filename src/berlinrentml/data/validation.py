"""Data validation utilities."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataValidator:
    """Validate dataset schema and quality."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues: List[str] = []

    def validate_schema(
        self, required_columns: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate that required columns exist.

        Args:
            required_columns: List of required column names

        Returns:
            (is_valid, list of missing columns)
        """
        if required_columns is None:
            return True, []

        missing = [col for col in required_columns if col not in self.df.columns]

        if missing:
            self.issues.append(f"Missing columns: {missing}")

        return len(missing) == 0, missing

    def check_missing_values(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Check for missing values.

        Args:
            threshold: Flag columns with missing percentage above this threshold

        Returns:
            Dictionary of column -> missing percentage for problematic columns
        """
        missing_pct = self.df.isnull().sum() / len(self.df)
        problematic = missing_pct[missing_pct > threshold].to_dict()

        if problematic:
            self.issues.append(
                f"Columns with >{threshold*100}% missing: {list(problematic.keys())}"
            )

        return problematic

    def check_duplicates(self, subset: Optional[List[str]] = None) -> int:
        """
        Check for duplicate rows.

        Args:
            subset: Columns to consider for identifying duplicates

        Returns:
            Number of duplicate rows
        """
        n_duplicates = self.df.duplicated(subset=subset).sum()

        if n_duplicates > 0:
            pct = n_duplicates / len(self.df) * 100
            self.issues.append(f"Found {n_duplicates:,} duplicate rows ({pct:.2f}%)")

        return n_duplicates

    def check_value_ranges(
        self, column: str, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> int:
        """
        Check if values are within expected range.

        Args:
            column: Column to check
            min_val: Minimum expected value
            max_val: Maximum expected value

        Returns:
            Number of out-of-range values
        """
        if column not in self.df.columns:
            return 0

        series = self.df[column].dropna()
        n_issues = 0

        if min_val is not None:
            n_below = (series < min_val).sum()
            if n_below > 0:
                self.issues.append(f"{column}: {n_below:,} values below {min_val}")
                n_issues += n_below

        if max_val is not None:
            n_above = (series > max_val).sum()
            if n_above > 0:
                self.issues.append(f"{column}: {n_above:,} values above {max_val}")
                n_issues += n_above

        return n_issues

    def check_target_variable(self, target_col: str) -> Dict[str, any]:
        """
        Validate target variable.

        Args:
            target_col: Name of target column

        Returns:
            Dictionary with target statistics
        """
        if target_col not in self.df.columns:
            self.issues.append(f"Target column '{target_col}' not found")
            return {}

        target = self.df[target_col].dropna()

        stats = {
            "count": len(target),
            "missing": self.df[target_col].isnull().sum(),
            "min": target.min(),
            "max": target.max(),
            "mean": target.mean(),
            "median": target.median(),
            "std": target.std(),
            "n_negative": (target < 0).sum(),
            "n_zero": (target == 0).sum(),
        }

        if stats["n_negative"] > 0:
            self.issues.append(f"Target has {stats['n_negative']} negative values")

        if stats["n_zero"] > 0:
            self.issues.append(f"Target has {stats['n_zero']} zero values")

        return stats

    def get_summary(self) -> str:
        """Get validation summary."""
        if not self.issues:
            return "✓ No validation issues found"

        return "Validation issues:\n" + "\n".join(f"  • {issue}" for issue in self.issues)


def validate_data(
    df: pd.DataFrame,
    target_col: str,
    required_columns: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Run full data validation.

    Args:
        df: DataFrame to validate
        target_col: Target column name
        required_columns: Required columns

    Returns:
        (is_valid, summary_message)
    """
    validator = DataValidator(df)

    # Check schema
    validator.validate_schema(required_columns)

    # Check missing values
    validator.check_missing_values(threshold=0.5)

    # Check duplicates
    validator.check_duplicates()

    # Check target
    validator.check_target_variable(target_col)

    summary = validator.get_summary()
    is_valid = len(validator.issues) == 0

    return is_valid, summary
