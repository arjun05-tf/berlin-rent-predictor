"""Model evaluation utilities."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    median_ae = np.median(np.abs(y_true - y_pred))

    # Percentage within thresholds
    errors = np.abs(y_true - y_pred)
    pct_within_50 = (errors <= 50).mean() * 100
    pct_within_100 = (errors <= 100).mean() * 100
    pct_within_200 = (errors <= 200).mean() * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_ae": median_ae,
        "pct_within_50": pct_within_50,
        "pct_within_100": pct_within_100,
        "pct_within_200": pct_within_200,
    }


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Print metrics in a readable format."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"MAE:              €{metrics['mae']:.2f}")
    print(f"RMSE:             €{metrics['rmse']:.2f}")
    print(f"R²:               {metrics['r2']:.4f}")
    print(f"Median AE:        €{metrics['median_ae']:.2f}")
    print(f"Within €50:       {metrics['pct_within_50']:.1f}%")
    print(f"Within €100:      {metrics['pct_within_100']:.1f}%")
    print(f"Within €200:      {metrics['pct_within_200']:.1f}%")
    print(f"{'='*60}\n")


def analyze_errors_by_bin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_values: pd.Series,
    bin_name: str = "Bin",
) -> pd.DataFrame:
    """
    Analyze errors by binned values.

    Args:
        y_true: True values
        y_pred: Predicted values
        bin_values: Values to bin by (e.g., price ranges, sizes)
        bin_name: Name of the bin dimension

    Returns:
        DataFrame with error analysis per bin
    """
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "error": y_true - y_pred,
            "abs_error": np.abs(y_true - y_pred),
            "bin": bin_values.values,
        }
    )

    summary = (
        df.groupby("bin")
        .agg(
            {
                "y_true": ["count", "mean"],
                "abs_error": ["mean", "median"],
                "error": ["mean", "std"],
            }
        )
        .round(2)
    )

    summary.columns = ["Count", "Mean Price", "MAE", "Median AE", "Mean Error", "Std Error"]
    summary.index.name = bin_name

    return summary


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
    """
    Analyze prediction residuals.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with residual statistics
    """
    residuals = y_true - y_pred

    return {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "min": np.min(residuals),
        "max": np.max(residuals),
        "q25": np.percentile(residuals, 25),
        "median": np.median(residuals),
        "q75": np.percentile(residuals, 75),
        "skewness": pd.Series(residuals).skew(),
        "kurtosis": pd.Series(residuals).kurtosis(),
    }


def compare_split_strategies(
    results: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Compare results across different split strategies.

    Args:
        results: Dictionary mapping strategy name to metrics dict

    Returns:
        DataFrame comparing strategies
    """
    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "Split Strategy"

    # Round for readability
    comparison_df = comparison_df.round(2)

    return comparison_df
