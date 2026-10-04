"""Model training utilities."""

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from xgboost import XGBRegressor

from berlinrentml.modeling.baselines import MeanPredictor, MedianPredictor


def get_model(model_name: str, **kwargs) -> Any:
    """
    Get model instance by name.

    Args:
        model_name: Model name
        **kwargs: Model parameters

    Returns:
        Model instance
    """
    models = {
        "mean": MeanPredictor,
        "median": MedianPredictor,
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elasticnet": ElasticNet,
        "random_forest": RandomForestRegressor,
        "lightgbm": LGBMRegressor,
        "xgboost": XGBRegressor,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    model_class = models[model_name]

    # Set default parameters for tree-based models
    if model_name == "random_forest" and not kwargs:
        kwargs = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 4, "random_state": 42, "n_jobs": -1}
    elif model_name == "lightgbm" and not kwargs:
        kwargs = {"n_estimators": 100, "max_depth": 20, "learning_rate": 0.1, "random_state": 42, "verbose": -1}
    elif model_name == "xgboost" and not kwargs:
        kwargs = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42}
    elif model_name == "ridge" and not kwargs:
        kwargs = {"alpha": 1.0, "random_state": 42}
    elif model_name == "lasso" and not kwargs:
        kwargs = {"alpha": 1.0, "random_state": 42}
    elif model_name == "elasticnet" and not kwargs:
        kwargs = {"alpha": 1.0, "l1_ratio": 0.5, "random_state": 42}

    return model_class(**kwargs)


def save_model_artifacts(
    model: Any,
    preprocessor: Any,
    feature_names: list,
    model_name: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Save model and preprocessing artifacts.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        feature_names: Input feature names
        model_name: Name for saving
        output_dir: Output directory

    Returns:
        Dictionary of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # Save model
    model_path = output_dir / f"{model_name}_model.joblib"
    joblib.dump(model, model_path)
    artifacts["model"] = model_path

    # Save preprocessor
    preprocessor_path = output_dir / f"{model_name}_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    artifacts["preprocessor"] = preprocessor_path

    # Save feature names
    features_path = output_dir / f"{model_name}_features.joblib"
    joblib.dump(feature_names, features_path)
    artifacts["features"] = features_path

    print(f"\nModel artifacts saved to {output_dir}/")
    for name, path in artifacts.items():
        print(f"  {name}: {path.name}")

    return artifacts


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
