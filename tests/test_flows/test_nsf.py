# -*- coding: utf-8 -*-
"""
Tests specific to the nueral spline flows
"""
import numpy as np
import pytest
import torch

from nessai.flows import NeuralSplineFlow

torch.set_grad_enabled(False)


@pytest.fixture
def uniform_flow():
    return NeuralSplineFlow(2, 2, 2, 1, base_distribution='uniform',
                            tail_bound=1.0, context_features=1)


def test_uniform_base_distibution(uniform_flow):
    """Test the log prob of the base dist"""
    x = torch.tensor([[-2, 0], [0, 0.5]])
    log_prob = uniform_flow.base_distribution_log_prob(x)
    np.testing.assert_array_almost_equal(log_prob.numpy(),
                                         np.array([-np.inf, 0.]))


def test_uniform_forward(uniform_flow):
    """Test the foward method with a uniform base distribution"""
    x = torch.rand(100, 2)
    c = torch.rand(100, 1)
    z, log_j = uniform_flow.forward(x, context=c)
    assert np.logical_and(z.numpy() > 0, z.numpy() < 1).all()


def test_uniform_inverse(uniform_flow):
    """Test the inverse method"""
    z = torch.rand(100, 2)
    c = torch.rand(100, 1)
    x, log_j = uniform_flow.inverse(z, context=c)
    assert np.logical_and(x.numpy() > 0, x.numpy() < 1).all()


def test_uniform_sample_and_log_prob(uniform_flow):
    """Test the sample and log_prob method"""
    c = torch.rand(100, 1)
    x, log_prob = uniform_flow.sample_and_log_prob(100, context=c)
    assert x.shape == (100, 2)


def test_invertilbity(uniform_flow):
    x = torch.tensor([[0.5, 0.8]])
    c = torch.tensor([[0.1]])

    z, _ = uniform_flow.forward(x, context=c)

    x_out, _ = uniform_flow.inverse(z, context=c)
