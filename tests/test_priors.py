"""
Test the prior functions.
"""
import numpy as np
import pytest

from nessai import priors


def test_uniform_in_bounds():
    """Test behaviour if x is out of bounds"""
    assert priors.log_uniform_prior(0) == 0.0


def test_uniform_out_of_bounds():
    """Test behaviour if x is out of bounds"""
    assert priors.log_uniform_prior(-2) == -np.inf


@pytest.mark.parametrize("k", [np.pi, 2 * np.pi])
def test_2d_cartesian_origin(k):
    """Test the value of the 2d Cartesian prior at the origin"""
    assert priors.log_2d_cartesian_prior(0, 0, k=k) == -np.log(k)


def test_2d_cartesian_sine_negative():
    """Test to ensure prior is 0 if y < 0"""
    x = np.array([[0.5], [-1.0]])
    assert priors.log_2d_cartesian_prior_sine(x[0], x[1]) == -np.inf


def test_2d_cartesian_sine_in_valid_range():
    """Test to ensure prior raises a runtime error if k != pi"""
    x = np.array([[0.5], [1.0]])
    with pytest.raises(RuntimeError) as excinfo:
        priors.log_2d_cartesian_prior_sine(x[0], x[1], k=2 * np.pi)
    assert "incompatible" in str(excinfo.value)


def test_3d_cartesian_at_origin():
    """Test the value of 3d Cartesian prior at the origin"""
    assert priors.log_3d_cartesian_prior(0, 0, 0) == (-1.5 * np.log(2 * np.pi))
