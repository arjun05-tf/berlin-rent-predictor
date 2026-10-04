"""Conformal interval tests."""

import numpy as np

from berlinrentml.modeling.uncertainty import ConformalInterval


def test_interval_covers_target_level():
    rng = np.random.default_rng(0)
    truth = rng.uniform(400, 2500, 5000)
    pred = truth * np.exp(rng.normal(0, 0.15, 5000))
    conf = ConformalInterval(alpha=0.1).fit(truth[:2500], pred[:2500])
    assert 0.87 <= conf.coverage(truth[2500:], pred[2500:]) <= 0.93


def test_interval_brackets_prediction():
    conf = ConformalInterval(0.1).fit(np.array([100.0, 200.0, 300.0]), np.array([110.0, 190.0, 330.0]))
    lo, hi = conf.predict(np.array([500.0]))
    assert lo[0] < 500 < hi[0]
