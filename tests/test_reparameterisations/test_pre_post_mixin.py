from unittest.mock import create_autospec

import numpy as np
import pytest

from nessai.reparameterisations.rescale import PrePostRescalingMixin
from nessai.utils.rescaling import rescaling_functions


@pytest.fixture
def reparam():
    return create_autospec(PrePostRescalingMixin)


def test_default_pre_rescaling(reparam):
    """Assert the default pre-rescaling is the identity"""
    x = np.array([1, 2, 3])
    expected_log_j = np.zeros(3)
    x_out, log_j = PrePostRescalingMixin.pre_rescaling(reparam, x)
    x_out_inv, log_j_inv = PrePostRescalingMixin.pre_rescaling_inv(reparam, x)

    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(x_out_inv, x)
    np.testing.assert_array_equal(log_j, expected_log_j)
    np.testing.assert_array_equal(log_j_inv, expected_log_j)


def test_default_post_rescaling(reparam):
    """Assert the default post-rescaling is the identity"""
    x = np.array([1, 2, 3])
    expected_log_j = np.zeros(3)
    x_out, log_j = PrePostRescalingMixin.post_rescaling(reparam, x)
    x_out_inv, log_j_inv = PrePostRescalingMixin.post_rescaling_inv(reparam, x)

    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(x_out_inv, x)
    np.testing.assert_array_equal(log_j, expected_log_j)
    np.testing.assert_array_equal(log_j_inv, expected_log_j)


def test_configure_pre_rescaling_none(reparam):
    """Test the configuration of the pre-rescaling if it is None"""
    PrePostRescalingMixin.configure_pre_rescaling(reparam, None)
    assert reparam.has_pre_rescaling is False


def test_configure_post_rescaling_none(reparam):
    """Test the configuration of the post-rescaling if it is None"""
    PrePostRescalingMixin.configure_post_rescaling(reparam, None)
    assert reparam.has_post_rescaling is False


def test_pre_rescaling_with_functions(reparam):
    """Assert that specifying functions works as intended"""
    rescaling = (np.exp, np.log)
    PrePostRescalingMixin.configure_pre_rescaling(reparam, rescaling)
    assert reparam.has_pre_rescaling is True
    assert reparam.pre_rescaling is np.exp
    assert reparam.pre_rescaling_inv is np.log


def test_post_rescaling_with_functions(reparam):
    """Assert that specifying functions works as intended"""
    rescaling = (np.exp, np.log)
    PrePostRescalingMixin.configure_post_rescaling(reparam, rescaling)
    assert reparam.has_post_rescaling is True
    assert reparam.has_prime_prior is False
    assert reparam.post_rescaling is np.exp
    assert reparam.post_rescaling_inv is np.log


@pytest.mark.parametrize("rescaling", ["logit", "inv_gaussian_cdf"])
def test_pre_rescaling_with_str(reparam, rescaling):
    """Assert that specifying a str works as intended"""
    PrePostRescalingMixin.configure_pre_rescaling(reparam, rescaling)
    assert reparam.has_pre_rescaling is True
    assert reparam.pre_rescaling is rescaling_functions[rescaling][0]
    assert reparam.pre_rescaling_inv is rescaling_functions[rescaling][1]


@pytest.mark.parametrize("rescaling", ["log", "logit"])
def test_post_rescaling_with_str(reparam, rescaling):
    """Assert that specifying a str works as intended"""
    reparam._update = False
    reparam.parameters = ["x"]

    PrePostRescalingMixin.configure_post_rescaling(reparam, rescaling)
    assert reparam.has_post_rescaling is True
    assert reparam.has_prime_prior is False
    assert reparam.post_rescaling is rescaling_functions[rescaling][0]
    assert reparam.post_rescaling_inv is rescaling_functions[rescaling][1]


def test_pre_rescaling_with_invalid_str(reparam):
    """Assert an error is raised if the rescaling is not recognised"""
    rescaling = "not_a_rescaling"
    with pytest.raises(RuntimeError) as excinfo:
        PrePostRescalingMixin.configure_pre_rescaling(reparam, rescaling)
    assert "Unknown rescaling function: not_a_rescaling" in str(excinfo.value)


def test_post_rescaling_with_invalid_str(reparam):
    """Assert an error is raised if the rescaling is not recognised"""
    rescaling = "not_a_rescaling"
    with pytest.raises(RuntimeError) as excinfo:
        PrePostRescalingMixin.configure_post_rescaling(reparam, rescaling)
    assert "Unknown rescaling function: not_a_rescaling" in str(excinfo.value)


def test_pre_rescaling_invalid_input(reparam):
    """Assert an error is raised if the input isn't a str or tuple"""
    with pytest.raises(RuntimeError) as excinfo:
        PrePostRescalingMixin.configure_pre_rescaling(reparam, (np.exp,))
    assert "Pre-rescaling must be a str or tuple" in str(excinfo.value)


def test_post_rescaling_invalid_input(reparam):
    """Assert an error is raised if the input isn't a str or tuple"""
    with pytest.raises(RuntimeError) as excinfo:
        PrePostRescalingMixin.configure_post_rescaling(reparam, (np.exp,))
    assert "Post-rescaling must be a str or tuple" in str(excinfo.value)
