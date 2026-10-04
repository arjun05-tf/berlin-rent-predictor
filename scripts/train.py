"""Main training script."""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from berlinrentml.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from berlinrentml.data import load_berlin_data
from berlinrentml.data.cleaning import clean_berlin_rental_data
from berlinrentml.data.validation import validate_data
from berlinrentml.features import FeatureEngineer, select_features
from berlinrentml.features.preprocessing import create_preprocessor
from berlinrentml.modeling.evaluation import calculate_metrics, print_metrics, compare_split_strategies
from berlinrentml.modeling.splitting import RandomSplit, GroupedSplit
from berlinrentml.modeling.training import get_model, save_model_artifacts


def main():
    """Main training pipeline."""
    print("="*80)
    print("BerlinRentML - Training Pipeline")
    print("="*80)

    # Step 1: Load data
    print("\n[1/9] Loading data...")
    try:
        df = load_berlin_data()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease download the dataset first:")
        print("1. Visit: https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany")
        print("2. Download the CSV file")
        print("3. Place it in: data/raw/")
        print("4. Rename to: immo_data.csv")
        return

    print(f"Loaded {len(df):,} Berlin listings")
    print(f"Columns: {df.columns.tolist()}")

    # Determine target column
    target_col = None
    if "baseRent" in df.columns:
        target_col = "baseRent"
    elif "totalRent" in df.columns:
        target_col = "totalRent"
        print("\n⚠️  Warning: Using totalRent as target. Ideally we want baseRent (cold rent).")
    else:
        print("❌ Error: No rent column found (baseRent or totalRent)")
        return

    print(f"Target variable: {target_col}")

    # Step 2: Validate data
    print("\n[2/9] Validating data...")
    is_valid, summary = validate_data(df, target_col=target_col)
    print(summary)

    # Step 3: Clean data
    print("\n[3/9] Cleaning data...")
    df_clean, cleaning_summary = clean_berlin_rental_data(df, target_col=target_col)
    print(cleaning_summary)
    print(f"Clean dataset: {len(df_clean):,} rows")

    # Step 4: Feature engineering
    print("\n[4/9] Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df_clean)
    print(f"Dataset with engineered features: {len(df_features):,} rows, {len(df_features.columns)} columns")

    # Step 5: Select features
    print("\n[5/9] Selecting features...")
    df_model = df_features.copy()

    # Select available features
    numeric_features = []
    for feat in ["livingSpace", "rooms", "floor", "building_age", "log_size", "rooms_per_m2", "amenity_score"]:
        if feat in df_model.columns:
            numeric_features.append(feat)

    categorical_features = []
    for feat in ["geo_plz", "geo_bln", "heatingType", "condition", "age_bin", "size_bin"]:
        if feat in df_model.columns:
            categorical_features.append(feat)

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    all_features = numeric_features + categorical_features
    df_model = df_model[all_features + [target_col]].copy()

    # Drop rows with missing target or key features
    df_model = df_model.dropna(subset=[target_col] + ["livingSpace", "rooms"])
    print(f"Final modeling dataset: {len(df_model):,} rows")

    # Save processed data
    processed_path = PROCESSED_DATA_DIR / "berlin_processed.csv"
    df_model.to_csv(processed_path, index=False)
    print(f"Saved processed data to: {processed_path}")

    # Step 6: Train/test splits
    print("\n[6/9] Creating train/test splits...")

    results = {}

    # Random split
    print("\n--- Random Split ---")
    random_splitter = RandomSplit(test_size=0.2, random_state=42)
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = random_splitter.split(df_model, target_col)
    print(f"Train: {len(X_train_rand):,}, Test: {len(X_test_rand):,}")

    # Grouped split (by postal code if available)
    if "geo_plz" in df_model.columns:
        print("\n--- Grouped Split (by postal code) ---")
        grouped_splitter = GroupedSplit(group_col="geo_plz", test_size=0.2, random_state=42)
        X_train_grouped, X_test_grouped, y_train_grouped, y_test_grouped = grouped_splitter.split(df_model, target_col)
        print(f"Train: {len(X_train_grouped):,}, Test: {len(X_test_grouped):,}")
    else:
        print("\n⚠️  Skipping grouped split (no postal code column)")
        X_train_grouped = X_train_rand
        X_test_grouped = X_test_rand
        y_train_grouped = y_train_rand
        y_test_grouped = y_test_rand

    # Step 7: Train models
    print("\n[7/9] Training models...")

    models_to_train = [
        "mean",
        "median",
        "linear",
        "ridge",
        "random_forest",
        "lightgbm",
    ]

    best_model_name = None
    best_model = None
    best_preprocessor = None
    best_score = float('inf')

    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        # Create preprocessor (fit on training data only!)
        preprocessor = create_preprocessor(
            X_train_rand,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        # Fit preprocessor on training data
        X_train_processed = preprocessor.fit_transform(X_train_rand)
        X_test_rand_processed = preprocessor.transform(X_test_rand)

        # Train model
        model = get_model(model_name)
        model.fit(X_train_processed, y_train_rand)

        # Evaluate on random split
        y_pred_rand = model.predict(X_test_rand_processed)
        metrics_rand = calculate_metrics(y_test_rand, y_pred_rand)
        print_metrics(metrics_rand, f"{model_name.upper()} - Random Split")

        results[f"{model_name}_random"] = metrics_rand

        # Evaluate on grouped split
        if "geo_plz" in df_model.columns:
            # Need to refit preprocessor on grouped training data
            preprocessor_grouped = create_preprocessor(
                X_train_grouped,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            X_train_grouped_processed = preprocessor_grouped.fit_transform(X_train_grouped)
            X_test_grouped_processed = preprocessor_grouped.transform(X_test_grouped)

            model_grouped = get_model(model_name)
            model_grouped.fit(X_train_grouped_processed, y_train_grouped)

            y_pred_grouped = model_grouped.predict(X_test_grouped_processed)
            metrics_grouped = calculate_metrics(y_test_grouped, y_pred_grouped)
            print_metrics(metrics_grouped, f"{model_name.upper()} - Grouped Split")

            results[f"{model_name}_grouped"] = metrics_grouped

            # Track best model based on grouped split MAE (geographic generalization)
            if metrics_grouped["mae"] < best_score:
                best_score = metrics_grouped["mae"]
                best_model_name = model_name
                best_model = model_grouped
                best_preprocessor = preprocessor_grouped
        else:
            # If no grouped split, use random split performance
            if metrics_rand["mae"] < best_score:
                best_score = metrics_rand["mae"]
                best_model_name = model_name
                best_model = model
                best_preprocessor = preprocessor

    # Step 8: Compare splits
    print("\n[8/9] Comparing split strategies...")
    comparison = compare_split_strategies(results)
    print("\n" + "="*80)
    print("SPLIT STRATEGY COMPARISON")
    print("="*80)
    print(comparison)
    print("="*80)

    # Step 9: Save best model
    print(f"\n[9/9] Saving final model: {best_model_name}")
    print(f"Selected based on grouped split MAE: €{best_score:.2f}")

    save_model_artifacts(
        model=best_model,
        preprocessor=best_preprocessor,
        feature_names=all_features,
        model_name="final_model",
        output_dir=MODELS_DIR,
    )

    print("\n" + "="*80)
    print("✅ Training pipeline complete!")
    print("="*80)
    print(f"\nFinal model: {best_model_name}")
    print(f"Best grouped split MAE: €{best_score:.2f}")
    print(f"\nModel artifacts saved to: {MODELS_DIR}/")
    print("\nNext steps:")
    print("1. Run error analysis: python scripts/analyze_errors.py")
    print("2. Run explainability: python scripts/explain_model.py")
    print("3. Start API: python src/berlinrentml/api/main.py")


if __name__ == "__main__":
    main()
