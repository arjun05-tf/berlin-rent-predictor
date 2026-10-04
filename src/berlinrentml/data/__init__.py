"""Data loading utilities."""

from pathlib import Path
from typing import Optional

import pandas as pd

from berlinrentml.config import RAW_DATA_DIR


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw dataset from CSV.

    Args:
        file_path: Path to CSV file. If None, searches for common names in RAW_DATA_DIR.

    Returns:
        DataFrame with raw data.

    Raises:
        FileNotFoundError: If no data file found.
    """
    if file_path is None:
        # Try common file names
        possible_names = [
            "immo_data.csv",
            "apartment-rental-offers-in-germany.csv",
            "apartments_data.csv",
        ]

        for name in possible_names:
            candidate = RAW_DATA_DIR / name
            if candidate.exists():
                file_path = candidate
                break

        if file_path is None:
            raise FileNotFoundError(
                f"No data file found in {RAW_DATA_DIR}. "
                f"Please download dataset and place in data/raw/. "
                f"See data/DATASET.md for instructions."
            )

    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

    return df


def filter_berlin(df: pd.DataFrame, region_col: str = "regio1") -> pd.DataFrame:
    """
    Filter dataset to Berlin only.

    Args:
        df: Raw DataFrame
        region_col: Column name containing region information

    Returns:
        DataFrame filtered to Berlin
    """
    if region_col not in df.columns:
        print(f"Warning: Column '{region_col}' not found. Available columns: {df.columns.tolist()}")
        return df

    berlin_df = df[df[region_col] == "Berlin"].copy()
    print(f"Filtered to Berlin: {len(berlin_df):,} rows ({len(berlin_df)/len(df)*100:.1f}%)")

    return berlin_df


def load_berlin_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and filter data to Berlin only.

    Args:
        file_path: Path to CSV file.

    Returns:
        DataFrame with Berlin data only.
    """
    df = load_raw_data(file_path)
    return filter_berlin(df)
