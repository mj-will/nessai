# -*- coding: utf-8 -*-
"""Tests related to plots in FlowProposal"""
import os
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from nessai.proposal import FlowProposal
from nessai.livepoint import numpy_array_to_live_points


@pytest.mark.parametrize("plot", [False, "all"])
def test_training_plots(proposal, tmpdir, plot):
    """Make sure traings plots are correctly produced"""
    proposal._plot_training = plot
    output = tmpdir.mkdir("test")

    names = ["x", "y"]
    prime_names = ["x_prime", "y_prime"]
    z = np.random.randn(10, 2)
    x = np.random.randn(10, 2)
    x_prime = x / 2
    proposal.training_data = numpy_array_to_live_points(x, names)
    proposal.training_data_prime = numpy_array_to_live_points(
        x_prime, prime_names
    )
    x_gen = numpy_array_to_live_points(x, names)
    x_prime_gen = numpy_array_to_live_points(x_prime, prime_names)

    # LogL will be populated before plotting
    for array in [
        proposal.training_data,
        proposal.training_data_prime,
        x_gen,
        x_prime_gen,
    ]:
        array["logL"] = 0.0

    proposal.dims = 2
    proposal.rescale_parameters = names
    proposal.rescaled_names = prime_names

    proposal.forward_pass = MagicMock(return_value=(z, None))
    proposal.backward_pass = MagicMock(return_value=(x_prime_gen, np.ones(10)))
    proposal.inverse_rescale = MagicMock(return_value=(x_gen, np.ones(10)))
    proposal.check_prior_bounds = lambda *args: args
    proposal.model = MagicMock()
    proposal.model.names = names

    FlowProposal._plot_training_data(proposal, output)

    assert os.path.exists(os.path.join(output, "x_samples.png")) is bool(plot)
    assert os.path.exists(os.path.join(output, "x_generated.png")) is bool(
        plot
    )
    assert os.path.exists(os.path.join(output, "x_prime_samples.png")) is bool(
        plot
    )
    assert os.path.exists(
        os.path.join(output, "x_prime_generated.png")
    ) is bool(plot)


def test_plot_pool_all(proposal):
    """Test for the plots that show the pool of samples"""
    proposal.output = "test"
    proposal._plot_pool = "all"
    proposal.populated_count = 0
    x = numpy_array_to_live_points(np.random.randn(10, 2), ["x", "y"])
    with patch("nessai.proposal.flowproposal.plot_live_points") as plot:
        FlowProposal.plot_pool(proposal, None, x)
    plot.assert_called_once_with(
        x, c="logL", filename=os.path.join("test", "pool_0.png")
    )


@pytest.mark.parametrize("alt_dist", [False, True])
def test_plot_pool_1d(proposal, tmpdir, alt_dist):
    """Test `plot_pool` when plotting is not 'all'.

    Test cases when there is an alternative base distribution is used and
    when one is not used.
    """
    output = tmpdir.mkdir("test_plot_pool_1d")
    proposal.output = output
    proposal._plot_pool = True
    proposal.populated_count = 0

    z = np.random.randn(10, 2)
    x = numpy_array_to_live_points(np.random.randn(10, 2), ["x", "y"])
    x["logL"] = np.random.randn(10)
    x["logP"] = np.random.randn(10)
    training_data = numpy_array_to_live_points(
        np.random.randn(10, 2), ["x", "y"]
    )
    training_data["logP"] = np.random.randn(10)
    training_data["logP"] = np.random.randn(10)
    proposal.training_data = training_data
    log_p = torch.arange(10)

    proposal.flow = MagicMock()
    proposal.flow.device = "cpu"
    if alt_dist:
        proposal.alt_dist = MagicMock()
        proposal.alt_dist.log_prob = MagicMock(return_value=log_p)
    else:
        proposal.flow.model.base_distribution_log_prob = MagicMock(
            return_value=log_p
        )
        proposal.alt_dist = None
    with patch("nessai.proposal.flowproposal.plot_1d_comparison") as plot:
        FlowProposal.plot_pool(proposal, z, x)

    plot.assert_called_once_with(
        training_data,
        x,
        labels=["live points", "pool"],
        filename=os.path.join(output, "pool_0.png"),
    )
    assert os.path.exists(os.path.join(output, "pool_0_log_q.png"))

    if alt_dist:
        proposal.alt_dist.log_prob.assert_called_once()
    else:
        proposal.flow.model.base_distribution_log_prob.assert_called_once()
