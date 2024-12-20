# -*- coding: utf-8 -*-
"""Test the augment proposal"""

from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest

from nessai.livepoint import empty_structured_array, numpy_array_to_live_points
from nessai.proposal import AugmentedFlowProposal


@pytest.fixture
def proposal(rng):
    return create_autospec(AugmentedFlowProposal, rng=rng)


@pytest.fixture
def x(rng):
    _x = np.concatenate(
        [rng.standard_normal((10, 2)), rng.standard_normal((10, 2))], axis=1
    )
    return numpy_array_to_live_points(_x, ["x", "y", "e_1", "e_2"])


@pytest.fixture
def x_prime(rng):
    _x = np.concatenate(
        [rng.standard_normal((10, 2)), rng.standard_normal((10, 2))], axis=1
    )
    return numpy_array_to_live_points(_x, ["x_prime", "y_prime", "e_1", "e_2"])


def test_init(model):
    """Test the init function"""
    AugmentedFlowProposal(model, poolsize=100)


def test_update_flow_config(proposal):
    """Test update flow config"""
    proposal.rescaled_dims = 4
    proposal.augment_dims = 2
    proposal.flow_config = dict(model_config={})
    with patch(
        "nessai.proposal.augmented.FlowProposal.update_flow_config"
    ) as mock:
        AugmentedFlowProposal.update_flow_config(proposal)
    mock.assert_called_once()
    mask = np.array([1, 1, -1, -1])
    np.testing.assert_array_equal(proposal.flow_config["mask"], mask)


@patch("nessai.proposal.FlowProposal.set_rescaling")
def test_set_rescaling(mock, proposal):
    """Test the set rescaling method"""
    proposal.parameters = ["x", "y"]
    proposal.prime_parameters = ["x_prime", "y_prime"]
    proposal.augment_dims = 2
    proposal.parameters_to_rescale = ["x", "y"]
    AugmentedFlowProposal.set_rescaling(proposal)

    assert proposal.parameters == ["x", "y", "e_0", "e_1"]
    assert proposal.prime_parameters == ["x_prime", "y_prime", "e_0", "e_1"]
    assert proposal.augment_parameters == ["e_0", "e_1"]
    mock.assert_called_once()


@pytest.mark.parametrize("generate", ["zeroes", "gaussian"])
@patch("numpy.zeros", return_value=np.ones(10))
def test_rescaling(mock_zeros, proposal, x, generate):
    """Test the rescaling method"""
    proposal._base_rescale = MagicMock(return_value=[x, np.ones(x.size)])
    proposal.augment_parameters = ["e_1"]
    proposal.augment_dims = 1
    proposal.rng = MagicMock()
    proposal.rng.standard_normal = MagicMock(return_value=np.arange(10))

    AugmentedFlowProposal._augmented_rescale(
        proposal, x, generate_augment=generate, test=True
    )

    proposal._base_rescale.assert_called_once_with(
        x, compute_radius=False, test=True
    )

    if generate == "zeroes":
        mock_zeros.assert_called_once_with(x.size)
    else:
        proposal.rng.standard_normal.assert_called_once_with(x.size)


@pytest.mark.parametrize("compute_radius", [False, True])
@patch("numpy.zeros", return_value=np.ones(10))
def test_rescaling_generate_none(mock_zeros, proposal, x, compute_radius):
    """Test the rescaling method with generate_augment=None"""
    proposal._base_rescale = MagicMock(return_value=[x, np.ones(x.size)])
    proposal.augment_parameters = ["e_1"]
    proposal.augment_dims = 1
    proposal.generate_augment = "zeros"
    proposal.rng = MagicMock()
    proposal.rng.standard_normal = MagicMock(return_value=np.arange(10))

    AugmentedFlowProposal._augmented_rescale(
        proposal,
        x,
        generate_augment=None,
        test=True,
        compute_radius=compute_radius,
    )

    proposal._base_rescale.assert_called_once_with(
        x, compute_radius=compute_radius, test=True
    )

    if not compute_radius:
        proposal.rng.standard_normal.assert_called_once_with(x.size)
    else:
        mock_zeros.assert_called_once_with(x.size)


def test_rescaling_generate_unknown(proposal, x):
    """Test the rescaling method with with an unknown method for generate"""
    proposal._base_rescale = MagicMock(return_value=[x, np.ones(x.size)])

    with pytest.raises(RuntimeError) as excinfo:
        AugmentedFlowProposal._augmented_rescale(
            proposal, x, generate_augment="ones"
        )
    assert "Unknown method" in str(excinfo.value)


def test_inverse_rescale(proposal, x_prime):
    proposal.augment_parameters = ["e_1", "e_2"]
    x = empty_structured_array(len(x_prime), ["x", "y", "e_1", "e_2"])
    x["x"] = x_prime["x_prime"].copy()
    x["y"] = x_prime["y_prime"].copy()
    proposal._base_inverse_rescale = MagicMock(
        return_value=[x, np.ones(x.size)]
    )
    x, log_J = AugmentedFlowProposal._augmented_inverse_rescale(
        proposal, x_prime
    )
    proposal._base_inverse_rescale.assert_called_once_with(
        x_prime, return_unit_hypercube=False
    )
    np.testing.assert_array_equal(x["e_1"], x_prime["e_1"])
    np.testing.assert_array_equal(x["e_2"], x_prime["e_2"])


def test_inverse_rescale_hypercube_error(proposal, x_prime):
    with pytest.raises(
        RuntimeError,
        match="Inverse rescaling with augmented parameters is not supported",
    ):
        AugmentedFlowProposal._augmented_inverse_rescale(
            proposal, x_prime, True
        )


@pytest.mark.parametrize("marg", [False, True])
def test_augmented_prior(marg, proposal, x):
    """Test the augmented prior with and without marginalistion"""
    log_prob = np.ones(len(x))
    proposal.marginalise_augment = marg
    proposal.augment_dist = MagicMock()
    proposal.augment_dist.logpdf = MagicMock(return_value=log_prob)
    proposal.augment_parameters = ["e_1", "e_2"]
    out = AugmentedFlowProposal.augmented_prior(proposal, x)
    if marg:
        np.testing.assert_array_equal(np.zeros(len(x)), out)
        proposal.augment_dist.logpdf.assert_not_called()
    else:
        assert out is log_prob
        proposal.augment_dist.logpdf.assert_called_once()
        np.testing.assert_array_equal(
            np.array([x["e_1"], x["e_2"]]).T,
            proposal.augment_dist.logpdf.call_args_list[0][0][0],
        )


@patch("nessai.proposal.flowproposal.FlowProposal.log_prior", return_value=1)
def test_log_prior(mock_prior, proposal, x):
    """Test the complete log prior"""
    proposal.augmented_prior = MagicMock(return_value=1)

    log_p = AugmentedFlowProposal.log_prior(proposal, x)

    mock_prior.assert_called_once_with(x)
    proposal.augmented_prior.assert_called_once_with(x)
    assert log_p == 2


@patch(
    "nessai.proposal.flowproposal.FlowProposal.x_prime_log_prior",
    return_value=1,
)
def test_prime_log_prior(mock_prior, proposal, x):
    """Test the complete prime log prior"""
    proposal.augmented_prior = MagicMock(return_value=1)

    log_p = AugmentedFlowProposal.x_prime_log_prior(proposal, x)

    mock_prior.assert_called_once_with(x)
    proposal.augmented_prior.assert_called_once_with(x)
    assert log_p == 2


def test_marginalise_augment(proposal, rng):
    """Test the marginalise augment function"""
    proposal.n_marg = 5
    proposal.augment_dims = 2
    x = np.concatenate([rng.standard_normal((3, 2)), np.zeros((3, 2))], axis=1)
    z = rng.standard_normal((15, 4))
    log_prob = rng.standard_normal(15)
    proposal.flow = MagicMock()
    proposal.flow.forward_and_log_prob = MagicMock(return_value=[z, log_prob])
    log_prob_out = AugmentedFlowProposal._marginalise_augment(proposal, x)

    assert len(log_prob_out) == 3


@pytest.mark.parametrize("log_p", [np.ones(2), np.array([-1, np.inf])])
@pytest.mark.parametrize("marg", [False, True])
@pytest.mark.parametrize("return_z", [False, True])
def test_backward_pass(proposal, model, log_p, marg, rng, return_z):
    """Test the backward pass method"""
    n = 2
    acc = int(np.isfinite(log_p).sum())
    x = rng.standard_normal((n, model.dims))
    z = rng.standard_normal((n, model.dims))
    proposal._marginalise_augment = MagicMock(return_value=log_p)
    proposal.inverse_rescale = MagicMock(
        side_effect=lambda a, return_unit_hypercube: (a, np.ones(a.size))
    )
    proposal.prime_parameters = model.names
    proposal.alt_dist = None
    proposal.check_prior_bounds = MagicMock(
        side_effect=lambda a, b, c: (a, b, c)
    )
    proposal.flow = MagicMock()
    proposal.flow.sample_and_log_prob = MagicMock(return_value=[x, log_p])

    proposal.marginalise_augment = marg

    out = AugmentedFlowProposal.backward_pass(proposal, z, return_z=return_z)

    if return_z:
        assert len(out) == 3
    else:
        assert len(out) == 2

    assert len(out[0]) == acc
    proposal.inverse_rescale.assert_called_once()
    proposal.flow.sample_and_log_prob.assert_called_once_with(
        z=z, alt_dist=None
    )

    assert proposal._marginalise_augment.called is marg


@pytest.mark.parametrize("return_z", [False, True])
def test_backward_pass_assertion_error(proposal, return_z):
    proposal.flow = MagicMock()
    proposal.flow.sample_and_log_prob = MagicMock(side_effect=AssertionError)
    z = np.ones((10, 2))
    out = AugmentedFlowProposal.backward_pass(proposal, z, return_z=return_z)
    if return_z:
        assert len(out) == 3
    else:
        assert len(out) == 2
    for a in out:
        assert len(a) == 0


@pytest.mark.integration_test
def test_w_default_rescaling(model, tmpdir):
    """Integration test to make sure augmented proposal works with the default
    rescaling method.
    """
    x = model.new_point(100)

    x_out = (
        2 * (x["x"] - model.bounds["x"][0]) / (np.ptp(model.bounds["x"])) - 1
    )
    y_out = (
        2 * (x["y"] - model.bounds["y"][0]) / (np.ptp(model.bounds["y"])) - 1
    )

    output = tmpdir.mkdir("testdir")
    proposal = AugmentedFlowProposal(
        model,
        output=output,
        poolsize=100,
        augment_dims=2,
        reparameterisations="rescaletobounds",
    )

    proposal.initialise()

    assert proposal.prime_parameters == ["x_prime", "y_prime", "e_0", "e_1"]

    x_prime, log_j = proposal.rescale(x)

    print(list(proposal._reparameterisation.values())[0].bounds)

    np.testing.assert_array_almost_equal(x_out, x_prime["x_prime"], decimal=15)
    np.testing.assert_array_almost_equal(y_out, x_prime["y_prime"], decimal=15)

    x_inv, log_j_inv = proposal.inverse_rescale(x_prime)

    np.testing.assert_array_almost_equal(log_j, -log_j_inv, decimal=15)
    np.testing.assert_array_almost_equal(x["x"], x_inv["x"], decimal=15)
    np.testing.assert_array_almost_equal(x["y"], x_inv["y"], decimal=15)


@pytest.mark.integration_test
def test_w_reparameterisation(model, tmpdir):
    """Integration test to make sure augmented proposal works with
    reparameterisaitons configured using the dictionary method.
    """
    x = model.new_point(100)
    reparameterisations = {
        "x": {"reparameterisation": "rescaletobounds", "update_bounds": False},
        "y": {"reparameterisation": "scale", "scale": 2.0},
    }

    x_out = (
        2 * (x["x"] - model.bounds["x"][0]) / (np.ptp(model.bounds["x"])) - 1
    )
    y_out = x["y"] / 2.0

    output = tmpdir.mkdir("testdir")
    proposal = AugmentedFlowProposal(
        model,
        output=output,
        poolsize=100,
        augment_dims=2,
        reparameterisations=reparameterisations,
    )

    proposal.initialise()

    assert proposal.prime_parameters == ["x_prime", "y_prime", "e_0", "e_1"]

    x_prime, log_j = proposal.rescale(x)

    np.testing.assert_array_almost_equal(x_out, x_prime["x_prime"], decimal=15)
    np.testing.assert_array_almost_equal(y_out, x_prime["y_prime"], decimal=15)

    x_inv, log_j_inv = proposal.inverse_rescale(x_prime)

    np.testing.assert_array_almost_equal(log_j, -log_j_inv, decimal=15)
    np.testing.assert_array_almost_equal(
        x["x"],
        x_inv["x"],
        decimal=15,
    )
    np.testing.assert_array_almost_equal(
        x["y"],
        x_inv["y"],
        decimal=15,
    )
