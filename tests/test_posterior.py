# -*- coding: utf-8 -*-
"""
Test functions related to drawing posterior samples
"""
import numpy as np

from nessai.livepoint import numpy_array_to_live_points
from nessai.posterior import compute_weights, draw_posterior_samples


def test_compute_weights():
    """Test computing the weights for set of likleihood values"""
    log_l = np.random.randn(20)

    log_z, log_w = compute_weights(log_l, 10)

    assert len(log_w) == len(log_l)
    assert np.isfinite(log_z)


def test_draw_posterior_samples():
    """Test drawing posterior samples."""
    samples = numpy_array_to_live_points(np.random.randn(20, 1), ['x'])
    samples['logL'] = np.log(np.random.rand(20))
    p = draw_posterior_samples(samples, 10)
    assert np.isin(p, samples).all()
