# -*- coding: utf-8 -*-
"""
Test functions related to training and using the flow.
"""
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal import FlowProposal
from nessai.utils.testing import assert_structured_arrays_equal


def test_reset_model_weights(proposal):
    """Test resetting model weights"""
    proposal.flow = MagicMock()
    proposal.flow.reset_model = MagicMock()
    FlowProposal.reset_model_weights(proposal, reset_permutations=True)
    proposal.flow.reset_model.assert_called_once_with(reset_permutations=True)


def test_train_not_initialised(proposal):
    """Assert an error is raised if the proposal is not initialised"""
    proposal.initialised = False
    with pytest.raises(
        RuntimeError, match=r"FlowProposal is not initialised."
    ):
        FlowProposal.train(proposal, [1, 2])


@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
def test_train_plot_false(mock_os_makedirs, proposal, model):
    """Test the train method"""
    x = model.new_point(2)
    x_prime = model.new_point(2)
    proposal.rescaled_names = model.names
    proposal.save_training_data = False
    proposal.training_count = 0
    proposal.populated = True
    proposal.flow = MagicMock()
    proposal.flow.train = MagicMock()
    proposal.check_state = MagicMock()
    proposal.rescale = MagicMock(return_value=(x_prime, np.zeros_like(x)))
    FlowProposal.train(proposal, x, plot=False)

    assert_structured_arrays_equal(x, proposal.training_data)
    proposal.check_state.assert_called_once_with(proposal.training_data)
    proposal.rescale.assert_called_once_with(x)
    assert proposal.populated is False
    assert proposal.training_count == 1
    proposal.flow.train.assert_called_once()


@pytest.mark.parametrize("n", [1, 10])
def test_forward_pass(proposal, model, n):
    """Test the forward pass method"""
    x = model.new_point(n)
    z = np.random.randn(n, model.dims)
    proposal.clip = False
    proposal.rescale = MagicMock(return_value=[x, 2 * np.ones(n)])
    proposal.rescaled_names = model.names
    proposal.flow = MagicMock()
    proposal.flow.forward_and_log_prob = MagicMock(
        return_value=[z, np.ones(n)]
    )

    z_out, log_p = FlowProposal.forward_pass(proposal, x, compute_radius=False)

    assert np.array_equal(z, z_out)
    assert np.array_equal(log_p, 3 * np.ones(n))
    proposal.rescale.assert_called_once_with(x, compute_radius=False)
    proposal.flow.forward_and_log_prob.assert_called_once()


@pytest.mark.parametrize("log_p", [np.ones(2), np.array([-1, np.inf])])
def test_backward_pass(proposal, model, log_p):
    """Test the forward pass method"""
    n = 2
    acc = int(np.isfinite(log_p).sum())
    x = np.random.randn(n, model.dims)
    z = np.random.randn(n, model.dims)
    proposal.inverse_rescale = MagicMock(
        side_effect=lambda a: (a, np.ones(a.size))
    )
    proposal.rescaled_names = model.names
    proposal.alt_dist = None
    proposal.check_prior_bounds = MagicMock(side_effect=lambda a, b: (a, b))
    proposal.flow = MagicMock()
    proposal.flow.sample_and_log_prob = MagicMock(return_value=[x, log_p])

    x_out, log_p = FlowProposal.backward_pass(proposal, z)

    assert len(x_out) == acc
    proposal.inverse_rescale.assert_called_once()
    proposal.flow.sample_and_log_prob.assert_called_once_with(
        z=z, alt_dist=None
    )


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize("plot", [True, False])
@pytest.mark.parametrize("plot_training", [True, False])
def test_training(proposal, tmpdir, save, plot, plot_training):
    """Test the training method"""
    output = tmpdir.mkdir("test")
    data = np.random.randn(10, 2)
    data_prime = data / 2
    x = numpy_array_to_live_points(data, ["x", "y"])
    x_prime = numpy_array_to_live_points(data_prime, ["x_prime", "y_prime"])
    log_j = np.ones(data.shape[0])

    proposal.initialised = True
    proposal.training_count = 0
    proposal.populated = True
    proposal._plot_training = plot_training
    proposal.save_training_data = save
    proposal.rescale_parameters = ["x"]
    proposal.rescaled_names = ["x_prime", "y_prime"]
    proposal.output = output

    proposal.check_state = MagicMock()
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.flow = MagicMock()
    proposal.flow.train = MagicMock()
    proposal._plot_training_data = MagicMock()

    with patch(
        "nessai.proposal.flowproposal.live_points_to_array",
        return_value=data_prime,
    ), patch("nessai.proposal.flowproposal.save_live_points") as mock_save:
        FlowProposal.train(proposal, x, plot=plot)

    assert_structured_arrays_equal(x, proposal.training_data)

    if save or (plot and plot_training):
        output = os.path.join(output, "training", "block_0", "")

    if save:
        mock_save.assert_called_once()

    if plot and plot_training:
        proposal._plot_training_data.assert_called_once_with(output)
    elif not plot or not plot_training:
        proposal._plot_training_data.assert_not_called()

    proposal.check_state.assert_called_once_with(proposal.training_data)
    proposal.rescale.assert_called_once_with(x)
    proposal.flow.train.assert_called_once_with(
        data_prime, output=output, plot=plot and plot_training
    )
    assert proposal.training_count == 1
    assert proposal.populated is False
