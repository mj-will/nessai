# -*- coding: utf-8 -*-
"""Basic tests for all of the included flows"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np

from nessai.flows import RealNVP, MaskedAutoregressiveFlow, NeuralSplineFlow

flows = [RealNVP, MaskedAutoregressiveFlow, NeuralSplineFlow]


@pytest.fixture(params=[2, 4])
def data_dim(request):
    return request.param


@pytest.fixture()
def conditional_features():
    return 1


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


@pytest.fixture(params=flows)
def conditional_flow(request, data_dim, conditional_features):
    return request.param(
        data_dim, 8, 2, 2, context_features=conditional_features
    ).eval()


@pytest.fixture(params=flows)
def flow_class(request):
    return request.param


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch_norm_between_layers=True),
        dict(batch_norm_within_layers=True),
        dict(activation=F.relu),
        dict(dropout_probability=0.5),
    ],
)
def test_init(flow_class, kwargs):
    """Test init method with common kwargs"""
    flow = flow_class(2, 2, 2, 2, **kwargs)
    x = torch.randn(10, 2)
    z, _ = flow.forward(x)
    assert z.shape == (10, 2)


def test_forward(flow, x, n, data_dim):
    """Test the forward pass of the flow"""
    with torch.inference_mode():
        z, _ = flow.forward(x)
    assert z.shape == (n, data_dim)


def test_inverse(flow, z, n, data_dim):
    """Test the inverse method"""
    with torch.inference_mode():
        x, _ = flow.inverse(z)
    assert x.shape == (n, data_dim)


def test_sample(flow, n, data_dim):
    """Test the sample method"""
    x = flow.sample(n)
    assert x.shape == (n, data_dim)


def test_log_prob(flow, x, n):
    """Test the log prob method"""
    with torch.inference_mode():
        log_prob = flow.log_prob(x)
    assert log_prob.shape == (n,)


def test_base_distribution_log_prob(flow, z, n):
    """Test the bast distribution"""
    with torch.inference_mode():
        log_prob = flow.base_distribution_log_prob(z)
    assert log_prob.shape == (n,)


def test_forward_and_log_prob(flow, x, n, data_dim):
    """
    Test the forward and log prob method.

    Tests to ensure method runs and that it agrees with using forward
    and log_prob separately
    """
    with torch.inference_mode():
        z, log_prob = flow.forward_and_log_prob(x)
        z_target, _ = flow.forward(x)
        log_prob_target = flow.log_prob(x)
    np.testing.assert_array_equal(z.numpy(), z_target.numpy())
    np.testing.assert_array_equal(log_prob.numpy(), log_prob_target.numpy())


@pytest.mark.flaky(reruns=5)
def test_sample_and_log_prob(flow, n, data_dim):
    """
    Assert that samples are drawn with correct shape and that the log
    prob is correct.
    """
    with torch.inference_mode():
        x, log_prob = flow.sample_and_log_prob(n)
        log_prob_target = flow.log_prob(x)
    assert x.shape == (n, data_dim)
    np.testing.assert_array_almost_equal(
        log_prob.numpy(), log_prob_target.numpy(), decimal=5
    )


@pytest.mark.flaky(reruns=5)
def test_invertibility(flow, x):
    """Test to ensure flows are invertible"""
    with torch.inference_mode():
        z, log_J = flow.forward(x)
        x_out, log_J_out = flow.inverse(z)

    np.testing.assert_array_almost_equal(x.numpy(), x_out.numpy(), decimal=5)
    np.testing.assert_array_almost_equal(
        log_J.numpy(), -log_J_out.numpy(), decimal=5
    )


@pytest.mark.flaky(reruns=5)
@pytest.mark.integration_test
def test_sample_and_log_prob_conditional(
    conditional_flow, n, data_dim, conditional_features
):
    """Test method for conditional flows."""
    c = torch.randn(n, conditional_features)
    with torch.inference_mode():
        x, log_prob = conditional_flow.sample_and_log_prob(n, context=c)
        log_prob_target = conditional_flow.log_prob(x, context=c)
    assert x.shape == (n, data_dim)
    np.testing.assert_array_almost_equal(
        log_prob.numpy(), log_prob_target.numpy(), decimal=5
    )
