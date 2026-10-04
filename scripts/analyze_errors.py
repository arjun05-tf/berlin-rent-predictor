"""Error analysis script."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from berlinrentml.config import MODELS_DIR, PROCESSED_DATA_DIR
from berlinrentml.features import FeatureEngineer
from berlinrentml.features.preprocessing import create_preprocessor
from berlinrentml.modeling.evaluation import (
    analyze_errors_by_bin,
    analyze_residuals,
    calculate_metrics,
)
from berlinrentml.modeling.splitting import RandomSplit
from berlinrentml.modeling.training import load_model_artifacts


def main():
    """Run error analysis."""
    print("=" * 80)
    print("BerlinRentML - Error Analysis")
    print("=" * 80)

    # Load processed data
    print("\n[1/5] Loading processed data...")
    data_path = PROCESSED_DATA_DIR / "berlin_processed.csv"

    if not data_path.exists():
        print(f"❌ Processed data not found: {data_path}")
        print("Run training script first: python scripts/train.py")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows")

    # Determine target column
    target_col = "baseRent" if "baseRent" in df.columns else "totalRent"

    # Split data
    print("\n[2/5] Splitting data...")
    splitter = RandomSplit(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(df, target_col)

    # Load model
    print("\n[3/5] Loading model...")
    try:
        model, preprocessor, feature_names = load_model_artifacts(
            model_name="final_model",
            model_dir=MODELS_DIR,
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Make predictions
    print("\n[4/5] Making predictions...")
    X_test_features = X_test[feature_names]
    X_test_processed = preprocessor.transform(X_test_features)
    y_pred = model.predict(X_test_processed)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"MAE:              €{metrics['mae']:.2f}")
    print(f"RMSE:             €{metrics['rmse']:.2f}")
    print(f"R²:               {metrics['r2']:.4f}")
    print(f"Median AE:        €{metrics['median_ae']:.2f}")
    print("=" * 80)

    # Error analysis
    print("\n[5/5] Analyzing errors...")

    # By price range
    print("\n--- Errors by Price Range ---")
    price_bins = pd.cut(
        y_test,
        bins=[0, 500, 1000, 1500, 10000],
        labels=["<€500", "€500-1000", "€1000-1500", ">€1500"],
    )
    price_analysis = analyze_errors_by_bin(y_test.values, y_pred, price_bins, "Price Range")
    print(price_analysis)

    # By apartment size
    if "livingSpace" in X_test.columns:
        print("\n--- Errors by Apartment Size ---")
        size_bins = pd.cut(
            X_test["livingSpace"],
            bins=[0, 40, 70, 100, 500],
            labels=["<40m²", "40-70m²", "70-100m²", ">100m²"],
        )
        size_analysis = analyze_errors_by_bin(y_test.values, y_pred, size_bins, "Size")
        print(size_analysis)

    # By district (if available)
    if "geo_bln" in X_test.columns:
        print("\n--- Errors by District (Top 10) ---")
        district_analysis = analyze_errors_by_bin(
            y_test.values, y_pred, X_test["geo_bln"], "District"
        )
        # Show top 10 by count
        print(district_analysis.nlargest(10, "Count"))

    # Residual analysis
    print("\n--- Residual Analysis ---")
    residuals = analyze_residuals(y_test.values, y_pred)
    print(f"Mean residual:     €{residuals['mean']:.2f}")
    print(f"Std residual:      €{residuals['std']:.2f}")
    print(f"Min residual:      €{residuals['min']:.2f}")
    print(f"Max residual:      €{residuals['max']:.2f}")
    print(f"Median residual:   €{residuals['median']:.2f}")
    print(f"Skewness:          {residuals['skewness']:.3f}")
    print(f"Kurtosis:          {residuals['kurtosis']:.3f}")

    # Worst predictions
    print("\n--- Worst 10 Predictions ---")
    errors = np.abs(y_test.values - y_pred)
    worst_indices = np.argsort(errors)[-10:][::-1]

    worst_df = pd.DataFrame(
        {
            "Actual": y_test.iloc[worst_indices].values,
            "Predicted": y_pred[worst_indices],
            "Error": errors[worst_indices],
            "Size (m²)": (
                X_test.iloc[worst_indices]["livingSpace"].values
                if "livingSpace" in X_test.columns
                else ["N/A"] * 10
            ),
            "Rooms": (
                X_test.iloc[worst_indices]["rooms"].values
                if "rooms" in X_test.columns
                else ["N/A"] * 10
            ),
        }
    )
    print(worst_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("✅ Error analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
