"""
Test the distance converter classes.
"""
import numpy as np
import pytest
from scipy import stats
from unittest.mock import create_autospec

from nessai.gw.utils import (
    NullDistanceConverter,
    PowerLawConverter
)


@pytest.fixture
def null_converter():
    create_autospec(NullDistanceConverter)


def test_null_converter_init(null_converter, caplog):
    """Test the null distance converter init"""
    NullDistanceConverter.__init__(null_converter, d_min=10)
    assert "Kwargs {'d_min': 10} will be ignored" in caplog.text


def test_null_converter_to_uniform(null_converter):
    """Test the null distance converter to uniform"""
    d = np.arange(10)
    d_out, log_j = \
        NullDistanceConverter.to_uniform_parameter(null_converter, d)
    np.testing.assert_array_equal(d_out, d)
    np.testing.assert_equal(log_j, 0)


def test_null_converter_from_uniform(null_converter):
    """Test the null distance converter from uniform"""
    d = np.arange(10)
    d_out, log_j = \
        NullDistanceConverter.from_uniform_parameter(null_converter, d)
    np.testing.assert_array_equal(d_out, d)
    np.testing.assert_equal(log_j, 0)


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
