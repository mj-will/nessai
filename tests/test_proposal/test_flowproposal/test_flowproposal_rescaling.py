# -*- coding: utf-8 -*-
"""Test methods related to reparameterisations"""
import numpy as np
from nessai.livepoint import get_dtype, numpy_array_to_live_points
from nessai.model import Model
from nessai.proposal import FlowProposal
from nessai.reparameterisations import (
    NullReparameterisation,
    RescaleToBounds,
    get_reparameterisation,
)
import pytest
from unittest.mock import MagicMock, Mock, call, patch


@pytest.fixture
def proposal(proposal):
    """Specific mocked proposal for reparameterisation tests"""
    proposal.use_default_reparameterisations = False
    proposal.reverse_reparameterisations = False
    proposal.model = MagicMock(spec=Model)
    return proposal


@pytest.fixture
def dummy_rc():
    """Dummy reparameteristation class"""
    m = MagicMock()
    m.__name__ = "DummyReparameterisation"
    return m


@pytest.fixture
def dummy_cmb_rc():
    """Dummy combined reparameteristation class"""
    m = MagicMock()
    m.add_reparameterisation = MagicMock()
    return m


def test_default_reparameterisation(proposal):
    """Test to make sure default reparameterisation does not cause errors
    for default proposal.
    """
    FlowProposal.add_default_reparameterisations(proposal)


@patch("nessai.proposal.flowproposal.get_reparameterisation")
def test_get_reparamaterisation(mocked_fn, proposal):
    """Make sure the underlying function is called"""
    FlowProposal.get_reparameterisation(proposal, "angle")
    mocked_fn.assert_called_once_with("angle")


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("use_default_reparameterisations", [False, True])
def test_configure_reparameterisations_dict(
    proposal,
    dummy_cmb_rc,
    dummy_rc,
    reverse_order,
    use_default_reparameterisations,
):
    """Test configuration for reparameterisations dictionary.

    Also tests to make sure boundary inversion is set and if the
    `reverse_reparameterisation` is correctly set.
    """
    dummy_rc.return_value = "r"
    # Need to add the parameters before hand to prevent a
    # NullReparameterisation from being added
    dummy_cmb_rc.parameters = ["x"]
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = MagicMock(
        return_value=(dummy_rc, {"boundary_inversion": True})
    )
    proposal.model = MagicMock()
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x"]
    proposal.reverse_reparameterisations = reverse_order
    proposal.use_default_reparameterisations = use_default_reparameterisations

    with patch(
        "nessai.proposal.flowproposal.CombinedReparameterisation",
        return_value=dummy_cmb_rc,
    ) as mocked_class:
        FlowProposal.configure_reparameterisations(
            proposal, {"x": {"reparameterisation": "default"}}
        )

    proposal.get_reparameterisation.assert_called_once_with("default")

    if use_default_reparameterisations:
        proposal.add_default_reparameterisations.assert_called_once()
    else:
        proposal.add_default_reparameterisations.assert_not_called()

    dummy_rc.assert_called_once_with(
        prior_bounds={"x": [-1, 1]}, parameters="x", boundary_inversion=True
    )
    mocked_class.assert_called_once_with(reverse_order=reverse_order)
    # fmt: off
    proposal._reparameterisation.add_reparameterisations \
        .assert_called_once_with("r")
    # fmt: on

    assert proposal.boundary_inversion is True
    assert proposal.names == ["x"]


@patch("nessai.proposal.flowproposal.CombinedReparameterisation")
def test_configure_reparameterisations_dict_w_params(
    mocked_class, proposal, dummy_rc, dummy_cmb_rc
):
    """Test configuration for reparameterisations dictionary with parameters.

    For example:

        {'x': {'reparmeterisation': 'default', 'parameters': 'y'}}

    This should add both x and y to the reparameterisation.
    """
    dummy_rc.return_value = "r"
    # Need to add the parameters before hand to prevent a
    # NullReparameterisation from being added
    dummy_cmb_rc.parameters = ["x", "y"]
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = MagicMock(
        return_value=(
            dummy_rc,
            {},
        )
    )
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]

    with patch(
        "nessai.proposal.flowproposal.CombinedReparameterisation",
        return_value=dummy_cmb_rc,
    ) as mocked_class:
        FlowProposal.configure_reparameterisations(
            proposal,
            {"x": {"reparameterisation": "default", "parameters": ["y"]}},
        )

    proposal.get_reparameterisation.assert_called_once_with("default")
    proposal.add_default_reparameterisations.assert_not_called()
    dummy_rc.assert_called_once_with(
        prior_bounds={"x": [-1, 1], "y": [-1, 1]},
        parameters=["y", "x"],
    )
    mocked_class.assert_called_once()
    # fmt: off
    proposal._reparameterisation.add_reparameterisations \
        .assert_called_once_with("r")
    # fmt: on

    assert proposal.names == ["x", "y"]


def test_configure_reparameterisations_requires_prime_prior(
    proposal, dummy_rc, dummy_cmb_rc
):
    """
    Test configuration that requires a prime prior but the prime prior is
    missing.
    """
    dummy_rc.return_value = "r"
    # Need to add the parameters before hand to prevent a
    # NullReparameterisation from being added
    dummy_cmb_rc.parameters = ["x", "y"]
    dummy_cmb_rc.has_prime_prior = False
    dummy_cmb_rc.requires_prime_prior = True
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = MagicMock(
        return_value=(
            dummy_rc,
            {},
        )
    )
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]

    with patch(
        "nessai.proposal.flowproposal.CombinedReparameterisation",
        return_value=dummy_cmb_rc,
    ), pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_reparameterisations(
            proposal,
            {"x": {"reparameterisation": "default", "parameters": ["y"]}},
        )

    assert "One or more reparameterisations require " in str(excinfo.value)


@patch("nessai.reparameterisations.CombinedReparameterisation")
def test_configure_reparameterisations_dict_missing(mocked_class, proposal):
    """
    Test configuration for reparameterisations dictionary when missing
    the reparameterisation for a parameter.

    This should raise a runtime error.
    """
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]

    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_reparameterisations(
            proposal, {"x": {"scale": 1.0}}
        )

    assert "No reparameterisation found for x" in str(excinfo.value)


def test_configure_reparameterisations_str(proposal):
    """Test configuration for reparameterisations dictionary from a str"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]
    proposal.fallback_reparameterisation = None
    FlowProposal.configure_reparameterisations(proposal, {"x": "default"})

    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.rescaled_names == ["x_prime", "y"]
    assert proposal.parameters_to_rescale == ["x"]
    assert proposal._reparameterisation.parameters == ["x", "y"]
    assert proposal._reparameterisation.prime_parameters == ["x_prime", "y"]


def test_configure_reparameterisations_dict_reparam(proposal):
    """Test configuration for reparameterisations dictionary"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]
    proposal.fallback_reparameterisation = None
    FlowProposal.configure_reparameterisations(
        proposal, {"default": {"parameters": ["x"]}}
    )

    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.rescaled_names == ["x_prime", "y"]
    assert proposal.parameters_to_rescale == ["x"]
    assert proposal._reparameterisation.parameters == ["x", "y"]
    assert proposal._reparameterisation.prime_parameters == ["x_prime", "y"]


@pytest.mark.parametrize(
    "parameters",
    [
        "x.*",
        [
            "x.*",
        ],
        ("x.*",),
    ],
)
def test_configure_reparameterisations_regex(proposal, parameters):
    """Test configuration for reparameterisations dictionary"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.names = ["x_0", "x_1", "y"]
    proposal.model.bounds = {"x_0": [-1, 1], "x_1": [-1, 1], "y": [-1, 1]}
    proposal.fallback_reparameterisation = None
    FlowProposal.configure_reparameterisations(
        proposal,
        {"z-score": {"parameters": parameters}},
    )

    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.rescaled_names == ["x_0_prime", "x_1_prime", "y"]
    assert proposal.parameters_to_rescale == ["x_0", "x_1"]
    assert proposal._reparameterisation.parameters == ["x_0", "x_1", "y"]
    assert proposal._reparameterisation.prime_parameters == [
        "x_0_prime",
        "x_1_prime",
        "y",
    ]


def test_configure_reparameterisations_none(proposal):
    """Test configuration when input is None"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]
    proposal.fallback_reparameterisation = None
    FlowProposal.configure_reparameterisations(proposal, None)
    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.rescaled_names == ["x", "y"]

    assert proposal.parameters_to_rescale == []
    assert proposal._reparameterisation.parameters == ["x", "y"]
    assert proposal._reparameterisation.prime_parameters == ["x", "y"]
    assert all(
        [
            isinstance(r, NullReparameterisation)
            for r in proposal._reparameterisation.reparameterisations.values()
        ]
    )


def test_configure_reparameterisations_fallback(proposal):
    """Test configuration when input is None"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]
    proposal.fallback_reparameterisation = "default"
    FlowProposal.configure_reparameterisations(proposal, None)
    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.rescaled_names == ["x_prime", "y_prime"]

    assert proposal.parameters_to_rescale == ["x", "y"]
    assert proposal._reparameterisation.parameters == ["x", "y"]
    assert proposal._reparameterisation.prime_parameters == [
        "x_prime",
        "y_prime",
    ]
    assert all(
        [
            isinstance(r, RescaleToBounds)
            for r in proposal._reparameterisation.reparameterisations.values()
        ]
    )


def test_configure_reparameterisations_incorrect_type(proposal):
    """Assert an error is raised when input is not a dictionary"""
    with pytest.raises(TypeError) as excinfo:
        FlowProposal.configure_reparameterisations(proposal, ["default"])
    assert "must be a dictionary" in str(excinfo.value)


def test_configure_reparameterisations_incorrect_config_type(proposal):
    """Assert an error is raised when the config for a key is not a dictionary
    or a known reparameterisation.
    """
    proposal.model.names = ["x"]
    with pytest.raises(TypeError) as excinfo:
        FlowProposal.configure_reparameterisations(proposal, {"x": ["a"]})
    assert "Unknown config type" in str(excinfo.value)


@pytest.mark.parametrize(
    "reparam",
    [{"z": {"reparameterisation": "sine"}}, {"sine": {"parameters": ["x"]}}],
)
def test_configure_reparameterisation_unknown(proposal, reparam):
    """
    Assert an error is raised if an unknown reparameterisation or parameters
    is passed.
    """
    proposal.model.names = ["x"]
    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_reparameterisations(proposal, reparam)
    assert "is not a parameter in the model or a known" in str(excinfo.value)


def test_configure_reparameterisation_no_parameters(proposal, dummy_rc):
    """Assert an error is raised if no parameters are specified"""
    proposal.model.names = ["x"]
    proposal.get_reparameterisation = MagicMock(
        return_value=(
            dummy_rc,
            {},
        )
    )
    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.configure_reparameterisations(
            proposal, {"default": {"update_bounds": True}}
        )
    assert "No parameters key" in str(excinfo.value)


def test_set_rescaling_with_model(proposal, model):
    """
    Test setting the rescaling when the model contains reparmaeterisations.
    """
    proposal.model = model
    proposal.model.reparameterisations = {"x": "default"}

    def update(self):
        proposal.names = model.names
        proposal.parameters_to_rescale = ["x"]
        proposal.rescaled_names = ["x_prime"]

    proposal.configure_reparameterisations = MagicMock()
    proposal.configure_reparameterisations.side_effect = update

    FlowProposal.set_rescaling(proposal)

    proposal.configure_reparameterisations.assert_called_once_with(
        {"x": "default"}
    )
    assert proposal.reparameterisations == {"x": "default"}
    assert proposal.rescaled_names == ["x_prime"]


def test_set_rescaling_with_reparameterisations(proposal, model):
    """
    Test setting the rescaling when a reparameterisations dict is defined.
    """
    proposal.model = model
    proposal.model.reparameterisations = None
    proposal.reparameterisations = {"x": "default"}

    def update(self):
        proposal.names = model.names
        proposal.parameters_to_rescale = ["x"]
        proposal.rescaled_names = ["x_prime"]

    proposal.configure_reparameterisations = MagicMock()
    proposal.configure_reparameterisations.side_effect = update

    FlowProposal.set_rescaling(proposal)

    proposal.configure_reparameterisations.assert_called_once_with(
        {"x": "default"}
    )
    assert proposal.reparameterisations == {"x": "default"}
    assert proposal.rescaled_names == ["x_prime"]


@pytest.mark.parametrize("rescale_parameters", [True, ["x"]])
def test_set_rescaling_parameters(proposal, model, rescale_parameters):
    """Test setting rescaling without reparameterisations."""

    rescale_bounds = [0, 1]
    boundary_inversion = True
    update_bounds = True
    inversion_type = "split"
    detect_edges = False
    detect_edges_kwargs = {}
    rescaled_names = ["x_prime"]

    _rescale_parameters = (
        model.names if rescale_parameters is True else rescale_parameters
    )

    def update(self):
        proposal.names = model.names
        proposal.parameters_to_rescale = _rescale_parameters
        proposal.rescaled_names = rescaled_names

    proposal.model = model
    proposal.model.reparameterisations = None
    proposal.reparameterisations = None
    proposal.configure_reparameterisations = MagicMock()
    proposal.configure_reparameterisations.side_effect = update

    proposal.rescale_parameters = rescale_parameters
    proposal.rescale_bounds = rescale_bounds
    proposal.update_bounds = update_bounds
    proposal.boundary_inversion = boundary_inversion
    proposal.inversion_type = inversion_type
    proposal.detect_edges = detect_edges
    proposal.detect_edges_kwargs = detect_edges_kwargs

    FlowProposal.set_rescaling(proposal)

    reparameterisations = {
        "rescaletobounds": {
            "parameters": _rescale_parameters,
            "rescale_bounds": rescale_bounds,
            "update_bounds": update_bounds,
            "boundary_inversion": boundary_inversion,
            "inversion_type": inversion_type,
            "detect_edges": detect_edges,
            "detect_edges_kwargs": detect_edges_kwargs,
        },
    }

    proposal.configure_reparameterisations.assert_called_with(
        reparameterisations
    )
    proposal.parameters_to_rescale == _rescale_parameters


@pytest.mark.parametrize("n", [1, 10])
def test_rescale(proposal, n):
    """Test rescaling when using reparameterisation dict"""
    x = numpy_array_to_live_points(np.random.randn(n, 2), ["x", "y"])
    x["logL"] = np.random.randn(n)
    x["logP"] = np.random.randn(n)
    x_prime = numpy_array_to_live_points(
        np.random.randn(n, 2), ["x_prime", "y_prime"]
    )
    proposal.x_prime_dtype = get_dtype(["x_prime", "y_prime"])
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.reparameterise = MagicMock(
        return_value=[x, x_prime, np.ones(x.size)]
    )

    x_prime_out, log_j = FlowProposal.rescale(
        proposal, x, compute_radius=False, test="lower"
    )

    np.testing.assert_array_equal(
        x_prime[["x_prime", "y_prime"]], x_prime_out[["x_prime", "y_prime"]]
    )
    np.testing.assert_array_equal(
        x[["logP", "logL"]], x_prime_out[["logP", "logL"]]
    )
    proposal._reparameterisation.reparameterise.assert_called_once()


@pytest.mark.parametrize("n", [1, 10])
def test_inverse_rescale(proposal, n):
    """Test rescaling when using reparameterisation dict"""
    x = numpy_array_to_live_points(np.random.randn(n, 2), ["x", "y"]).squeeze()
    x_prime = numpy_array_to_live_points(
        np.random.randn(n, 2), ["x_prime", "y_prime"]
    )
    x_prime["logL"] = np.random.randn(n)
    x_prime["logP"] = np.random.randn(n)
    proposal.x_dtype = get_dtype(["x", "y"])
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.inverse_reparameterise = MagicMock(
        return_value=[x, x_prime, np.ones(x.size)]
    )

    x_out, log_j = FlowProposal.inverse_rescale(proposal, x_prime)

    np.testing.assert_array_equal(x[["x", "y"]], x_out[["x", "y"]])
    np.testing.assert_array_equal(
        x_prime[["logP", "logL"]], x_out[["logP", "logL"]]
    )
    proposal._reparameterisation.inverse_reparameterise.assert_called_once()


@pytest.mark.parametrize("has_inversion", [False, True])
def test_verify_rescaling(proposal, has_inversion):
    """Test the method that tests the rescaling at runtime

    Checks both normal parameters and non-sampling parameters (e.g logL)
    """
    x = np.array(
        [(1, np.nan), (2, np.nan)], dtype=[("x", "f8"), ("logL", "f8")]
    )
    x_prime = x["x"] / 2
    log_j = np.array([-2, -2])
    x_out = x.copy()
    log_j_inv = np.array([2, 2])

    if has_inversion:
        x_prime = np.concatenate([x_prime, x_prime])
        log_j = np.concatenate([log_j, log_j])
        log_j_inv = np.concatenate([log_j_inv, log_j_inv])
        x_out = np.concatenate([x_out, x_out])

    proposal.model.new_point = MagicMock(return_value=x)
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.inverse_rescale = MagicMock(return_value=(x_out, log_j_inv))
    proposal.check_state = MagicMock()
    proposal.rescaling_set = True

    FlowProposal.verify_rescaling(proposal)

    proposal.check_state.assert_has_calls(4 * [call(x)])
    # Should call 4 different test cases
    calls = [
        call(x, test="lower"),
        call(x, test="upper"),
        call(x, test=False),
        call(x, test=None),
    ]
    proposal.rescale.assert_has_calls(calls)
    proposal.inverse_rescale.assert_has_calls(4 * [call(x_prime)])


@pytest.mark.parametrize("has_inversion", [False, True])
def test_verify_rescaling_invertible_error(proposal, has_inversion):
    """Assert an error is raised if the rescaling is not invertible"""
    x = np.array([[1], [2]], dtype=[("x", "f8")])
    x_prime = x["x"] / 2
    log_j = np.array([-2, -2])
    x_out = x.copy()[::-1]
    log_j_inv = np.array([2, 2])

    if has_inversion:
        x_prime = np.concatenate([x_prime, x_prime])
        log_j = np.concatenate([log_j, log_j])
        log_j_inv = np.concatenate([log_j_inv, log_j_inv])
        x_out = np.concatenate([x_out, x_out])

    proposal.model.new_point = MagicMock(return_value=x)
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.inverse_rescale = MagicMock(return_value=(x_out, log_j_inv))
    proposal.rescaling_set = True

    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.verify_rescaling(proposal)
    assert "Rescaling is not invertible for x" in str(excinfo.value)


@pytest.mark.parametrize("has_inversion", [False, True])
def test_verify_rescaling_invertible_error_non_sampling(
    proposal, has_inversion
):
    """Assert an error is raised a non-sampler parameter changes"""
    x = np.array([(1, 3), (2, np.nan)], dtype=[("x", "f8"), ("logL", "f8")])
    x_prime = x["x"] / 2
    log_j = np.array([-2, -2])
    x_out = x.copy()
    log_j_inv = np.array([2, 2])

    if has_inversion:
        x_prime = np.concatenate([x_prime, x_prime])
        log_j = np.concatenate([log_j, log_j])
        log_j_inv = np.concatenate([log_j_inv, log_j_inv])
        x_out = np.concatenate([x_out, x_out])
    # Change the last element, this will test both cases
    x_out["logL"][-1] = 4

    proposal.model.new_point = MagicMock(return_value=x)
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.inverse_rescale = MagicMock(return_value=(x_out, log_j_inv))
    proposal.rescaling_set = True

    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.verify_rescaling(proposal)
    assert "Non-sampling parameter logL changed" in str(excinfo.value)


@pytest.mark.parametrize("has_inversion", [False, True])
def test_verify_rescaling_jacobian_error(proposal, has_inversion):
    """Assert an error is raised if the Jacobian is not invertible"""
    x = np.array([[1], [2]], dtype=[("x", "f8")])
    x_prime = x["x"] / 2
    log_j = np.array([-2, -2])
    x_out = x.copy()
    log_j_inv = np.array([2, 1])

    if has_inversion:
        x_prime = np.concatenate([x_prime, x_prime])
        log_j = np.concatenate([log_j, log_j])
        log_j_inv = np.concatenate([log_j_inv, log_j_inv])
        x_out = np.concatenate([x_out, x_out])

    proposal.model.new_point = MagicMock(return_value=x)
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.inverse_rescale = MagicMock(return_value=(x_out, log_j_inv))
    proposal.rescaling_set = True

    with pytest.raises(RuntimeError) as excinfo:
        FlowProposal.verify_rescaling(proposal)
    assert "Rescaling Jacobian is not invertible" in str(excinfo.value)


def test_verify_rescaling_rescaling_not_set(proposal):
    """Assert an error is raised if the rescaling is not set"""
    proposal.rescaling_set = False
    with pytest.raises(RuntimeError, match=r"Rescaling must be set .*"):
        FlowProposal.verify_rescaling(proposal)


def test_check_state_update(proposal):
    """Assert the update method is called"""
    x = numpy_array_to_live_points(np.random.randn(10, 2), ["x", "y"])
    proposal._reparameterisation = Mock()
    proposal._reparameterisation.update = MagicMock()
    FlowProposal.check_state(proposal, x)
    proposal._reparameterisation.update.assert_called_once_with(x)
