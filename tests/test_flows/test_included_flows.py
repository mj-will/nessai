import pytest
import torch
import numpy as np

from nessai.flows import (
    FlexibleRealNVP,
    MaskedAutoregressiveFlow,
    NeuralSplineFlow
    )

flows = [FlexibleRealNVP, MaskedAutoregressiveFlow, NeuralSplineFlow]


@pytest.fixture()
def data_dim():
    return 2


@pytest.fixture()
def n():
    return 1000


@pytest.fixture()
def x(n, data_dim):
    return torch.randn(n, data_dim).float()


@pytest.fixture()
def z(n, data_dim):
    return torch.randn(n, data_dim).float()


@pytest.fixture(params=flows)
def flow(request, data_dim):
    return request.param(data_dim, 8, 2, 2).eval()


def test_forward(flow, x, n, data_dim):
    """Test the foward pass of the flow"""
    with torch.no_grad():
        z, _ = flow.forward(x)
    assert z.shape == (n, data_dim)


def test_inverse(flow, z, n, data_dim):
    """Test the inverse method"""
    with torch.no_grad():
        x, _ = flow.inverse(z)
    assert x.shape == (n, data_dim)


def test_sample(flow, n, data_dim):
    """Test the sample method"""
    x = flow.sample(n)
    assert x.shape == (n, data_dim)


def test_log_prob(flow, x, n):
    """Test the log prob method"""
    with torch.no_grad():
        log_prob = flow.log_prob(x)
    assert log_prob.shape == (n,)


def test_base_distribution_log_prob(flow, z, n):
    """Test the bast distribution"""
    with torch.no_grad():
        log_prob = flow.base_distribution_log_prob(z)
    assert log_prob.shape == (n,)


def test_forward_and_log_prob(flow, x, n, data_dim):
    """
    Test the forward and log prob method.

    Tests to ensure method runs and that it agrees with using forward
    and log_prob seperately
    """
    with torch.no_grad():
        z, log_prob = flow.forward_and_log_prob(x)
        z_target, _ = flow.forward(x)
        log_prob_target = flow.log_prob(x)
    np.testing.assert_array_equal(z.numpy(), z_target.numpy())
    np.testing.assert_array_equal(log_prob.numpy(), log_prob_target.numpy())


@pytest.mark.flaky(run=10)
def test_sample_and_log_prob(flow, n, data_dim):
    """
    Assert that samples are drawn with correct shape and that the log
    prob is correct.
    """
    with torch.no_grad():
        x, log_prob = flow.sample_and_log_prob(n)
        log_prob_target = flow.log_prob(x)
    assert x.shape == (n, data_dim)
    np.testing.assert_array_almost_equal(
        log_prob.numpy(), log_prob_target.numpy(), decimal=5)


def test_invertibility(flow, x):
    """Test to ensure flows are invertible"""
    with torch.no_grad():
        z, log_J = flow.forward(x)
        x_out, log_J_out = flow.inverse(z)

    np.testing.assert_array_almost_equal(x.numpy(), x_out.numpy())
    np.testing.assert_array_almost_equal(
        log_J.numpy(), -log_J_out.numpy(), decimal=5)
