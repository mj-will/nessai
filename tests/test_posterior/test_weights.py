# -*- coding: utf-8 -*-
"""
Test functions related to computing posterior weights.
"""
import numpy as np
from nessai.posterior.weights import compute_weights


def test_compute_weights():
    """Test computing the weights for set of likelihood values"""
    log_l = np.random.randn(20)

    log_z, log_w = compute_weights(log_l, 10)

    assert len(log_w) == len(log_l)
    assert np.isfinite(log_z)
