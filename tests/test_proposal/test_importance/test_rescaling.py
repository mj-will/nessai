"""Test the rescaling functions in ImportanceFlowProposal"""

from unittest.mock import MagicMock, patch

from nessai import config
from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal.importance import ImportanceFlowProposal as IFP
import numpy as np
import pytest


def test_verify_rescaling_pass(ifp, x_prime):
    n = 10
    names = ["x", "y"]
    x = numpy_array_to_live_points(np.random.randn(n, len(names)), names)
    # Make sure there are NaNs for the Nan check
    assert np.isnan(x["logL"]).all()
    log_j = np.random.rand(n)
    x_re = x.copy()
    x_re["x"] += 1e-10 * np.random.randn(n)
    log_j_inv = -log_j
    log_j_inv += 1e-10 * np.random.randn(n)

    ifp.model.sample_unit_hypercube = MagicMock(return_value=x)
    ifp.rescale = MagicMock(return_value=(x_prime, log_j))
    ifp.inverse_rescale = MagicMock(return_value=(x_re, log_j_inv))

    IFP.verify_rescaling(ifp)

    ifp.rescale.assert_called_once_with(x)
    ifp.inverse_rescale.assert_called_once_with(x_prime)


def test_verify_rescaling_fail(ifp, x_prime):
    n = 10
    names = ["x", "y"]
    x = numpy_array_to_live_points(np.random.randn(n, len(names)), names)
    # Make sure there are NaNs for the Nan check
    assert np.isnan(x["logL"]).all()
    log_j = np.random.rand(n)
    x_re = x.copy()
    x_re["x"] += 1e-6 * np.random.randn(n)
    log_j_inv = -log_j
    log_j_inv += 1e-10 * np.random.randn(n)

    ifp.model.sample_unit_hypercube = MagicMock(return_value=x)
    ifp.rescale = MagicMock(return_value=(x_prime, log_j))
    ifp.inverse_rescale = MagicMock(return_value=(x_re, log_j_inv))

    with pytest.raises(RuntimeError, match=r"Rescaling is not invertible."):
        IFP.verify_rescaling(ifp)


def test_verify_rescaling_fail_jacobian(ifp, x_prime):
    n = 10
    names = ["x", "y"]
    x = numpy_array_to_live_points(np.random.randn(n, len(names)), names)
    # Make sure there are NaNs for the Nan check
    assert np.isnan(x["logL"]).all()
    log_j = np.random.rand(n)
    x_re = x.copy()
    x_re["x"] += 1e-10 * np.random.randn(n)
    log_j_inv = -log_j
    log_j_inv += 1e-7 * np.random.randn(n)

    ifp.model.sample_unit_hypercube = MagicMock(return_value=x)
    ifp.rescale = MagicMock(return_value=(x_prime, log_j))
    ifp.inverse_rescale = MagicMock(return_value=(x_re, log_j_inv))

    with pytest.raises(RuntimeError, match=r"Forward and inverse"):
        IFP.verify_rescaling(ifp)


def test_to_prime_logit(ifp, x_array, x_prime):
    """Assert logit is called and the Jacobian is correct"""
    ifp.reparameterisation = "logit"
    log_j = np.random.randn(*x_array.shape)
    log_j_exp = log_j.sum(axis=1)
    with patch(
        "nessai.proposal.importance.logit", return_value=(x_prime, log_j)
    ) as mock_logit:
        x_prime_out, log_j_out = IFP.to_prime(ifp, x_array)

    mock_logit.assert_called_once_with(x_array, eps=config.general.eps)
    assert x_prime_out is x_prime
    np.testing.assert_array_equal(log_j_out, log_j_exp)


def test_to_prime_none(ifp, x_array):
    """Assert no operation is applied and the log-Jacobian is 0"""
    ifp.reparameterisation = None
    x_prime_out, log_j_out = IFP.to_prime(ifp, x_array)
    np.testing.assert_array_equal(x_prime_out, x_array)
    assert np.all(log_j_out == 0.0)


def test_from_prime_sigmoid(ifp, x_array, x_prime):
    """Assert sigmoid is called and the Jacobian is correct"""
    ifp.reparameterisation = "logit"
    log_j = np.random.randn(*x_array.shape)
    log_j_exp = log_j.sum(axis=1)
    with patch(
        "nessai.proposal.importance.sigmoid", return_value=(x_array, log_j)
    ) as mock_sigmoid:
        x_out, log_j_out = IFP.from_prime(ifp, x_prime)

    mock_sigmoid.assert_called_once_with(x_prime)
    assert x_out is x_array
    np.testing.assert_array_equal(log_j_out, log_j_exp)


def test_from_prime_none(ifp, x_prime):
    """Assert no operation is applied and the log-Jacobian is 0"""
    ifp.reparameterisation = None
    x_out, log_j_out = IFP.from_prime(ifp, x_prime)
    np.testing.assert_array_equal(x_out, x_prime)
    assert np.all(log_j_out == 0.0)


def test_rescale(ifp, x, x_prime, log_j, model):
    """Assert rescale calls the correct functions in the correct order"""
    names = model.names
    x_array = np.random.randn(len(x), len(names))
    x = numpy_array_to_live_points(x_array, names)

    ifp.model.names = names
    ifp.to_prime = MagicMock(return_value=(x_prime, log_j))

    with patch(
        "nessai.proposal.importance.live_points_to_array",
        return_value=x_array,
    ) as mock_to_array:
        x_prime_out, log_j_out = IFP.rescale(ifp, x)

    mock_to_array.assert_called_once_with(x, names)
    ifp.to_prime.assert_called_once_with(x_array)

    assert x_prime_out is x_prime
    assert log_j_out is log_j


@pytest.mark.parametrize("clip", [True, False])
def test_inverse_rescale(ifp, x, x_prime, log_j, clip, model):
    """
    Assert inverse_rescale calls the correct functions in the correct order
    """
    names = model.names
    x_array = np.random.randn(len(x), len(names))

    ifp.clip = clip
    ifp.model.names = names
    ifp.from_prime = MagicMock(return_value=(x_array, log_j))

    with patch(
        "nessai.proposal.importance.numpy_array_to_live_points",
        return_value=x,
    ) as mock_to_array, patch("numpy.clip", return_value=x_array) as mock_clip:
        x_out, log_j_out = IFP.inverse_rescale(ifp, x_prime)

    ifp.from_prime.assert_called_once_with(x_prime)
    if clip:
        mock_clip.assert_called_once_with(x_array, 0.0, 1.0)
    else:
        mock_clip.assert_not_called()
    mock_to_array.assert_called_once_with(x_array, names)

    assert x_out is x
    assert log_j_out is log_j


def test_invalid_reparameterisation_to_prime(ifp, x):
    """Assert an invalid reparameterisation raises an error"""
    ifp.reparameterisation = "invalid"
    with pytest.raises(
        ValueError, match=r"Unknown reparameterisation: 'invalid'"
    ):
        IFP.to_prime(ifp, x)


def test_invalid_reparameterisation_from_prime(ifp, x_prime):
    """Assert an invalid reparameterisation raises an error"""
    ifp.reparameterisation = "invalid"
    with pytest.raises(
        ValueError, match=r"Unknown reparameterisation: 'invalid'"
    ):
        IFP.from_prime(ifp, x_prime)
