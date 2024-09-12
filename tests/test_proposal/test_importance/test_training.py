"""Test training functions in ImportanceFlowProposal"""

import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal.importance import ImportanceFlowProposal as IFP
from nessai.utils.testing import assert_structured_arrays_equal


@pytest.fixture
def ifp(ifp, tmp_path):
    ifp.output = tmp_path / "test_train"
    ifp.level_count = 2
    ifp.weighted_kl = False
    ifp._reset_flow = False
    ifp._weights = {-1: 0.25, 0: 0.25, 1: 0.25, 2: 0.25}
    return ifp


@pytest.fixture(params=[True, False])
def reset(request):
    return request.param


def test_train_basic(ifp, x, x_prime, reset):
    """Test the main training components"""
    ifp.rescale = Mock(return_value=(x_prime, None))
    ifp._reset_flow = reset
    ifp.plot_training = True

    ifp.flow = Mock()
    ifp.flow.models = [object(), object(), object()]

    def add_flow(reset=False):
        ifp.flow.models.append(object())

    ifp.flow.add_new_flow = Mock(side_effect=add_flow)

    IFP.train(ifp, samples=x, plot=False, test=True)

    # Number of levels should be level index +1
    assert len(ifp.flow.models) == (ifp.level_count + 1)

    ifp.flow.train.assert_called_once_with(
        x_prime,
        weights=None,
        output=os.path.join(ifp.output, "level_3", ""),
        plot=True,
        test=True,  # Make sure kwargs are passed
    )

    ifp.flow.add_new_flow.assert_called_once_with(reset=reset)
    assert len(ifp._weights) == (ifp.level_count + 2)
    assert ifp._weights[ifp.level_count] is np.nan


@patch("nessai.proposal.importance.plot_live_points")
@patch("nessai.proposal.importance.plot_1d_comparison")
@patch("nessai.proposal.importance.plot_histogram")
@pytest.mark.usefixtures("ins_parameters")
def test_train_plotting(
    mock_hist,
    mock_plot_1d,
    mock_plot_live_points,
    ifp,
    x,
    x_prime,
    model,
):
    """Test plotting in training function"""

    names = model.names
    level_output = os.path.join(ifp.output, "level_3", "")

    n = 10
    weights = np.random.rand(n)
    prime_samples = np.random.randn(n, len(names))
    samples = numpy_array_to_live_points(prime_samples, names)
    log_prob = np.random.rand(n)
    log_j = np.random.randn(n)

    samples["logQ"] = log_prob

    ifp.rescale = Mock(return_value=(x_prime, None))
    ifp.inverse_rescale = Mock(return_value=(samples, log_j))
    ifp.flow = MagicMock()
    ifp.flow.sample_and_log_prob = Mock(return_value=(prime_samples, log_prob))

    IFP.train(ifp, samples=x, plot=True, weights=weights)

    mock_hist.assert_called_once()
    np.testing.assert_array_equal(
        mock_hist.call_args[0][0], weights / np.sum(weights)
    )
    assert mock_hist.call_args_list[0][1]["filename"] == os.path.join(
        level_output, "training_weights.png"
    )

    mock_plot_1d.assert_called_once_with(
        x_prime,
        convert_to_live_points=True,
        filename=os.path.join(level_output, "prime_training_data.png"),
    )

    assert_structured_arrays_equal(
        mock_plot_live_points.call_args_list[0][0],
        x,
    )

    assert_structured_arrays_equal(
        mock_plot_live_points.call_args_list[1][0],
        samples,
    )

    print(mock_plot_live_points.call_args_list[0])
    assert mock_plot_live_points.call_args_list[0][1][
        "filename"
    ] == os.path.join(level_output, "training_data.png")
    assert mock_plot_live_points.call_args_list[1][1][
        "filename"
    ] == os.path.join(level_output, "generated_samples.png")


@pytest.mark.parametrize(
    "weights, weighted_kl", [[np.random.rand(10), False], [None, True]]
)
@pytest.mark.usefixtures("ins_parameters")
def test_train_weights(ifp, x, x_prime, weights, weighted_kl):
    x["logW"] = np.log(np.random.rand(x.size))

    ifp.rescale = Mock(return_value=(x_prime, None))
    ifp.flow = MagicMock()
    ifp.weighted_kl = weighted_kl
    ifp.plot_training = False

    if weights is not None:
        expected_weights = weights / np.sum(weights)
    else:
        expected_weights = np.exp(x["logW"]) / np.sum(np.exp(x["logW"]))

    IFP.train(ifp, samples=x, weights=weights, plot=False)

    actual_weights = ifp.flow.train.call_args[1]["weights"]
    np.testing.assert_almost_equal(np.sum(actual_weights), 1.0, decimal=10)
    np.testing.assert_array_almost_equal(actual_weights, expected_weights)


def test_train_weights_nan(ifp, x, x_prime):
    ifp.rescale = Mock(return_value=(x_prime, None))
    ifp.flow = MagicMock()
    ifp.weighted_kl = False
    ifp.plot_training = False
    weights = np.ones(x.size)
    weights[0] = np.nan
    with pytest.raises(ValueError, match=r"Weights contain NaN\(s\)"):
        IFP.train(ifp, samples=x, weights=weights, plot=False)


@pytest.mark.integration_test
@pytest.mark.usefixtures("ins_parameters")
def test_training_and_prob(model, tmp_path):
    ifp = IFP(
        model,
        output=tmp_path / "test_training_and_prob",
        weighted_kl=False,
    )
    ifp.initialise()
    weights = {-1: 1.0}
    for i in range(4):
        ifp.train(model.new_point(10), max_epochs=2)
        weights = {j - 1: 1 / (i + 2) for j in range(i + 2)}
        ifp.update_proposal_weights(weights)
        x, _ = ifp.draw(10)
    log_Q, log_q = ifp.compute_meta_proposal_samples(x)

    assert len(log_Q) == 10
    assert log_q.shape == (10, 5)
