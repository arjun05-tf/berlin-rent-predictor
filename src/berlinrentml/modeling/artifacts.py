"""Load saved model artifacts (light imports, safe for serverless)."""

from pathlib import Path
from typing import Any, Tuple

import joblib


def load_model_artifacts(
    model_name: str,
    model_dir: Path,
) -> Tuple[Any, Any, list]:
    """
    Load model and preprocessing artifacts.

    Args:
        model_name: Name of saved model
        model_dir: Directory containing artifacts

    Returns:
        (model, preprocessor, feature_names)
    """
    model_dir = Path(model_dir)

    model_path = model_dir / f"{model_name}_model.joblib"
    preprocessor_path = model_dir / f"{model_name}_preprocessor.joblib"
    features_path = model_dir / f"{model_name}_features.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_names = joblib.load(features_path)

    return model, preprocessor, feature_names


def load_conformal(model_dir: Path):
    """Load the conformal interval if one was saved, else None."""
    path = Path(model_dir) / "final_model_conformal.joblib"
    return joblib.load(path) if path.exists() else None
