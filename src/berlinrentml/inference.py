"""Shared inference: raw listing dict to predicted rent."""

import pandas as pd

from berlinrentml.features import FeatureEngineer


def predict_rent(data: dict, model, preprocessor, feature_names: list) -> float:
    """Engineer features for one listing and return the predicted base rent in euros."""
    df = FeatureEngineer().engineer_features(pd.DataFrame([data]))
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = None  # preprocessor imputes missing values
    return round(float(model.predict(preprocessor.transform(df[feature_names]))[0]), 2)
