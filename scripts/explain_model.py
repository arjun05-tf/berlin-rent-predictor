"""Model explainability with SHAP."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from berlinrentml.config import MODELS_DIR, PROCESSED_DATA_DIR
from berlinrentml.modeling.splitting import RandomSplit
from berlinrentml.modeling.training import load_model_artifacts

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def main():
    """Run SHAP explainability analysis."""
    print("=" * 80)
    print("BerlinRentML - Model Explainability (SHAP)")
    print("=" * 80)

    if not SHAP_AVAILABLE:
        print("\n❌ SHAP not installed. Install with: pip install shap")
        return

    # Load processed data
    print("\n[1/4] Loading processed data...")
    data_path = PROCESSED_DATA_DIR / "berlin_processed.csv"

    if not data_path.exists():
        print(f"❌ Processed data not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    target_col = "baseRent" if "baseRent" in df.columns else "totalRent"

    # Split data
    splitter = RandomSplit(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(df, target_col)

    # Load model
    print("\n[2/4] Loading model...")
    try:
        model, preprocessor, feature_names = load_model_artifacts(
            model_name="final_model",
            model_dir=MODELS_DIR,
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print(f"Model type: {model.__class__.__name__}")

    # Prepare data
    X_train_features = X_train[feature_names]
    X_test_features = X_test[feature_names]

    X_train_processed = preprocessor.transform(X_train_features)
    X_test_processed = preprocessor.transform(X_test_features)

    # Get processed feature names
    processed_feature_names = preprocessor.get_feature_names_out()

    # Create SHAP explainer
    print("\n[3/4] Computing SHAP values...")
    print("This may take a few minutes...")

    # Use a sample for faster computation
    sample_size = min(100, len(X_train_processed))
    X_train_sample = X_train_processed[:sample_size]

    # Choose explainer based on model type
    model_name = model.__class__.__name__.lower()

    if "tree" in model_name or "forest" in model_name or "lgbm" in model_name or "xgb" in model_name:
        # Tree-based explainer (fast)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_processed[:100])
    else:
        # Kernel explainer (slower, model-agnostic)
        explainer = shap.KernelExplainer(model.predict, X_train_sample)
        shap_values = explainer.shap_values(X_test_processed[:100])

    # Global feature importance
    print("\n[4/4] Analyzing feature importance...")
    print("\n" + "=" * 80)
    print("GLOBAL FEATURE IMPORTANCE (Top 15)")
    print("=" * 80)

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create feature importance dataframe
    importance_df = pd.DataFrame(
        {
            "feature": processed_feature_names,
            "importance": mean_abs_shap,
        }
    )

    importance_df = importance_df.sort_values("importance", ascending=False)

    # Show top 15
    print(importance_df.head(15).to_string(index=False))

    # Example predictions
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS WITH SHAP EXPLANATIONS")
    print("=" * 80)

    n_examples = 3
    for i in range(n_examples):
        print(f"\n--- Example {i+1} ---")
        actual = y_test.iloc[i]
        predicted = model.predict(X_test_processed[i:i+1])[0]

        print(f"Actual rent:     €{actual:.2f}")
        print(f"Predicted rent:  €{predicted:.2f}")
        print(f"Error:           €{actual - predicted:.2f}")

        # Get top contributing features
        example_shap = shap_values[i]
        top_features_idx = np.argsort(np.abs(example_shap))[-5:][::-1]

        print("\nTop 5 contributing features:")
        for idx in top_features_idx:
            feature_name = processed_feature_names[idx]
            shap_value = example_shap[idx]
            print(f"  {feature_name:30s}: {shap_value:+8.2f}")

    print("\n" + "=" * 80)
    print("✅ Explainability analysis complete!")
    print("=" * 80)
    print("\nNote: SHAP values show the impact of each feature on the prediction.")
    print("Positive values increase the prediction, negative values decrease it.")


if __name__ == "__main__":
    main()
