"""Model training utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
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


def tune_hyperparameters(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: Optional[np.ndarray] = None,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune hyperparameters using GridSearchCV.

    Args:
        model_name: Model name
        X_train: Training features
        y_train: Training target
        groups: Group labels for GroupKFold (if using spatial CV)
        cv: Number of CV folds
        random_state: Random seed

    Returns:
        (best_model, best_params)
    """
    # Define parameter grids
    param_grids = {
        "ridge": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "lasso": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "elasticnet": {
            "alpha": [0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.9],
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [2, 4, 8],
        },
        "lightgbm": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, 30],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 50, 100],
        },
        "xgboost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.05, 0.1],
        },
    }

    if model_name not in param_grids:
        print(f"No parameter grid defined for {model_name}, returning default model")
        return get_model(model_name), {}

    base_model = get_model(model_name)
    param_grid = param_grids[model_name]

    # Use GroupKFold if groups provided (for spatial CV)
    if groups is not None:
        cv_splitter = GroupKFold(n_splits=min(cv, len(np.unique(groups))))
        cv_splits = cv_splitter.split(X_train, y_train, groups)
    else:
        cv_splits = cv

    print(f"Tuning {model_name} with {len(param_grid)} parameters...")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_splits if groups is not None else cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train, groups=groups if groups is not None else None)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV MAE: €{-grid_search.best_score_:.2f}")

    return grid_search.best_estimator_, grid_search.best_params_


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
