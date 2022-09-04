# -*- coding: utf-8 -*-
"""Tests for the optimisation utilities."""
from unittest.mock import patch

import numpy as np
import pytest

from nessai.livepoint import numpy_array_to_live_points
from nessai.utils.testing import assert_structured_arrays_equal
from nessai.utils.optimise import optimise_meta_proposal_weights


@pytest.mark.usefixtures("ins_parameters")
def test_optimise_meta_weights():
    """Assert correct method from scipy is called"""
    n = 100
    n_its = 10
    samples = numpy_array_to_live_points(np.random.randn(n, 2), ["x0", "x1"])
    samples["logL"] = np.random.randn(n)
    samples["logQ"] = np.random.randn(n)
    samples["logW"] = -samples["logW"]
    samples["it"] = np.random.randint(-1, n_its, size=n)
    log_q = np.random.randn(n, n_its + 1)

    input_samples = samples.copy()

    with patch("nessai.utils.optimise.minimize") as mock:
        optimise_meta_proposal_weights(samples, log_q=log_q)

    mock.assert_called_once()

    # Make sure inputs are unchanged
    assert_structured_arrays_equal(samples, input_samples)
