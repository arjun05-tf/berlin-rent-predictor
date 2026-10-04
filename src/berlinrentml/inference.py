"""Shared inference: raw listing dict to predicted rent."""

import pandas as pd

from berlinrentml.features import FeatureEngineer


def predict_rent(data: dict, model, preprocessor, feature_names: list, interval=None):
    """Predicted base rent in euros for one listing; with `interval`, returns (rent, low, high)."""
    df = FeatureEngineer().engineer_features(pd.DataFrame([data]))
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = None  # preprocessor imputes missing values
    pred = model.predict(preprocessor.transform(df[feature_names]))
    rent = round(float(pred[0]), 2)
    if interval is None:
        return rent
    lo, hi = interval.predict(pred)
    return rent, round(float(lo[0]), 2), round(float(hi[0]), 2)
