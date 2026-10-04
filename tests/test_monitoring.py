"""Drift metric tests."""

import numpy as np

from berlinrentml.monitoring import psi


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.05


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(size=5000), rng.normal(1.5, 1, 5000)) > 0.25
