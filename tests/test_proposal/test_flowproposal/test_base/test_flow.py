"""
Test functions related to training and using the flow.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal.flowproposal.base import BaseFlowProposal
from nessai.utils.testing import assert_structured_arrays_equal


def test_reset_model_weights(proposal):
    """Test resetting model weights"""
    proposal.flow = MagicMock()
    proposal.flow.reset_model = MagicMock()
    BaseFlowProposal.reset_model_weights(proposal, reset_permutations=True)
    proposal.flow.reset_model.assert_called_once_with(reset_permutations=True)


def test_train_not_initialised(proposal):
    """Assert an error is raised if the proposal is not initialised"""
    proposal.initialised = False
    proposal.__name__ = "BaseFlowProposal"
    with pytest.raises(
        RuntimeError, match=r"BaseFlowProposal is not initialised."
    ):
        BaseFlowProposal.train(proposal, [1, 2])


@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
def test_train_plot_false(mock_os_makedirs, proposal, model):
    """Test the train method"""
    x = model.new_point(2)
    x_prime = model.new_point(2)
    proposal.prime_parameters = model.names
    proposal.save_training_data = False
    proposal.training_count = 0
    proposal.populated = True
    proposal.flow = MagicMock()
    proposal.flow.train = MagicMock()
    proposal.check_state = MagicMock()
    proposal.rescale = MagicMock(return_value=(x_prime, np.zeros_like(x)))
    BaseFlowProposal.train(proposal, x, plot=False)

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
    proposal.prime_parameters = model.names
    proposal.flow = MagicMock()
    proposal.flow.forward_and_log_prob = MagicMock(
        return_value=[z, np.ones(n)]
    )

    z_out, log_p = BaseFlowProposal.forward_pass(
        proposal, x, compute_radius=False
    )

    assert np.array_equal(z, z_out)
    assert np.array_equal(log_p, 3 * np.ones(n))
    proposal.rescale.assert_called_once_with(x, compute_radius=False)
    proposal.flow.forward_and_log_prob.assert_called_once()


@pytest.mark.parametrize("rescale", [True, False])
def test_backward_pass(proposal, rescale):
    """Test the backward pass method"""
    z = np.random.randn(10, 2)
    x_prime = np.random.randn(*z.shape)
    x = numpy_array_to_live_points(
        np.random.randn(*x_prime.shape),
        ["x", "y"],
    )
    log_j = np.random.rand(z.shape[0])
    proposal.flow = MagicMock()
    # log_j has in place operations
    proposal.flow.inverse = MagicMock(return_value=(x_prime, log_j.copy()))
    proposal.prime_parameters = ["x_prime", "y_prime"]

    x_rescale = numpy_array_to_live_points(
        np.random.randn(*x_prime.shape),
        ["x", "y"],
    )
    log_j_rescale = np.random.rand(x_prime.shape[0])
    proposal.inverse_rescale = MagicMock(
        return_value=(x_rescale, log_j_rescale)
    )

    with patch(
        "nessai.proposal.flowproposal.base.numpy_array_to_live_points",
        return_value=x,
    ) as mock:
        x_out, log_j_out = BaseFlowProposal.backward_pass(
            proposal, z, rescale=rescale, test=True
        )

    proposal.flow.inverse.assert_called_once_with(z)
    mock.assert_called_once()
    np.testing.assert_array_equal(mock.call_args[0][0], x_prime)
    assert mock.call_args[0][1] == proposal.prime_parameters

    if not rescale:
        proposal.inverse_rescale.assert_not_called()
        assert x_out is x
        np.testing.assert_array_equal(log_j_out, log_j)
    else:
        proposal.inverse_rescale.assert_called_once_with(x, test=True)
        assert x_out is x_rescale
        np.testing.assert_array_equal(log_j_out, log_j_rescale + log_j)


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize("plot", [True, False])
@pytest.mark.parametrize("plot_training", [True, False])
def test_training(proposal, tmp_path, save, plot, plot_training):
    """Test the training method"""
    output = tmp_path / "test"
    output.mkdir()
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
    proposal.prime_parameters = ["x_prime", "y_prime"]
    proposal.output = output

    proposal.check_state = MagicMock()
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.flow = MagicMock()
    proposal.flow.train = MagicMock()
    proposal._plot_training_data = MagicMock()

    with (
        patch(
            "nessai.proposal.flowproposal.base.live_points_to_array",
            return_value=data_prime,
        ),
        patch(
            "nessai.proposal.flowproposal.base.save_live_points"
        ) as mock_save,
    ):
        BaseFlowProposal.train(proposal, x, plot=plot)

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
