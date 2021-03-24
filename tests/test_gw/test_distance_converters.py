"""
Test the distance converter classes.
"""
import numpy as np
import pytest
from scipy import stats

from nessai.gw.utils import (
    PowerLawConverter,
    )


@pytest.mark.parametrize('power', [1, 2, 3, 4])
@pytest.mark.flaky(run=5)
def test_power_law_converter_distribution(power):
    """
    Check that the distribution of resulting samples is uniform when
    converting from a power law.
    """
    c = PowerLawConverter(power, scale=1)
    x = stats.powerlaw(power + 1).rvs(size=10000)
    y, _ = c.to_uniform_parameter(x)
    d, p = stats.kstest(y, 'uniform')
    assert p >= 0.05


@pytest.mark.parametrize('power', [1, 2, 3, 4])
def test_power_law_converter_inversion(power):
    """
    Check that the power law inversion is invertible
    """
    c = PowerLawConverter(power, scale=1)
    x = stats.powerlaw(power + 1).rvs(size=1000)
    y, _ = c.to_uniform_parameter(x)
    x_out, _ = c.from_uniform_parameter(y)
    np.testing.assert_array_almost_equal(x, x_out)
