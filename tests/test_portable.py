"""Portable forest must reproduce LightGBM predictions."""

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor

from berlinrentml.modeling.portable import PortableLGBM


@pytest.mark.parametrize("log_target", [False, True])
def test_matches_lightgbm(log_target):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6))
    X[rng.random(X.shape) < 0.05] = np.nan
    y = 500 + 100 * np.nan_to_num(X[:, 0]) + 50 * np.nan_to_num(X[:, 1]) ** 2 + rng.normal(0, 5, 500)
    reg = LGBMRegressor(n_estimators=40, num_leaves=15, random_state=0, verbose=-1)
    model = TransformedTargetRegressor(reg, func=np.log1p, inverse_func=np.expm1) if log_target else reg
    model.fit(X, y)
    np.testing.assert_allclose(PortableLGBM.from_model(model).predict(X), model.predict(X), rtol=1e-6)
