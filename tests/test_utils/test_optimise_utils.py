# -*- coding: utf-8 -*-
"""Tests for the optimisation utilities."""
from collections import namedtuple
from unittest.mock import patch

import numpy as np
import pytest

from nessai.livepoint import numpy_array_to_live_points
from nessai.utils.testing import assert_structured_arrays_equal
from nessai.utils.optimise import optimise_meta_proposal_weights

n_its = 10


@pytest.mark.usefixtures("ins_parameters")
@pytest.mark.parametrize("initial_weights", [None, np.arange(n_its)])
def test_optimise_meta_weights(initial_weights):
    """Assert correct method from scipy is called"""
    n = 100
    samples = numpy_array_to_live_points(np.random.randn(n, 2), ["x0", "x1"])
    samples["logL"] = np.random.randn(n)
    samples["logQ"] = np.random.randn(n)
    samples["logW"] = -samples["logW"]
    samples["it"] = np.random.randint(-1, n_its, size=n)
    log_q = np.random.randn(n, n_its + 1)

    input_samples = samples.copy()

    expected = np.random.rand(n_its + 1)

    Result = namedtuple("Result", ["x"])
    result = Result(x=expected)

    def check_loss(loss, *args, **kwargs):
        value = loss(np.random.rand(n_its + 1))
        assert np.isfinite(value)
        return result

    with patch(
        "nessai.utils.optimise.minimize", side_effect=check_loss
    ) as mock:
        out = optimise_meta_proposal_weights(
            samples, log_q=log_q, initial_weights=initial_weights
        )

    mock.assert_called_once()
    np.testing.assert_array_equal(out, result.x)

    # Make sure inputs are unchanged
    assert_structured_arrays_equal(samples, input_samples)
