"""Test methods related to reparameterisations"""

from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from nessai.livepoint import (
    get_dtype,
    live_points_to_array,
    numpy_array_to_live_points,
)
from nessai.model import Model
from nessai.proposal.flowproposal.base import BaseFlowProposal
from nessai.reparameterisations import (
    NullReparameterisation,
    Reparameterisation,
    ReparameterisationError,
    RescaleToBounds,
    get_reparameterisation,
)


@pytest.fixture
def proposal(proposal):
    """Specific mocked proposal for reparameterisation tests"""
    proposal.use_default_reparameterisations = False
    proposal.reverse_reparameterisations = False
    proposal.model = MagicMock(spec=Model)
    proposal._set_parameter_order = (
        BaseFlowProposal._set_parameter_order.__get__(
            proposal, BaseFlowProposal
        )
    )
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


class DummyFlowProposal(BaseFlowProposal):
    def populate(self, worst_point, n_samples=10000):
        raise NotImplementedError(
            "This is a dummy proposal and does not implement populate."
        )


def make_test_proposal(reparameterisations, rng):
    """Construct a concrete proposal with a toy four-parameter model."""
    model = MagicMock(spec=Model)
    model.names = ["a", "b", "c", "d"]
    model.bounds = {n: [-1.0, 1.0] for n in model.names}
    model.reparameterisations = None

    proposal = DummyFlowProposal(
        model,
        rng=rng,
        poolsize=10,
        reparameterisations=reparameterisations,
        fallback_reparameterisation=None,
    )
    proposal.get_reparameterisation = MagicMock(
        side_effect=get_reparameterisation
    )
    proposal.set_rescaling()
    return proposal


def test_default_reparameterisation(proposal):
    """Test to make sure default reparameterisation does not cause errors
    for default proposal.
    """
    BaseFlowProposal.add_default_reparameterisations(proposal)


@patch("nessai.proposal.flowproposal.base.get_reparameterisation")
def test_get_reparamaterisation(mocked_fn, proposal):
    """Make sure the underlying function is called"""
    BaseFlowProposal.get_reparameterisation(proposal, "angle")
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
    proposal.map_to_unit_hypercube = False
    proposal.model = MagicMock()
    proposal.model.bounds = {"x": [-1, 1]}
    proposal.model.names = ["x"]
    proposal.fallback_reparameterisations = None
    proposal.reverse_reparameterisations = reverse_order
    proposal.use_default_reparameterisations = use_default_reparameterisations
    proposal.prior_bounds = proposal.model.bounds

    with patch(
        "nessai.proposal.flowproposal.base.CombinedReparameterisation",
        return_value=dummy_cmb_rc,
    ) as mocked_class:
        BaseFlowProposal.configure_reparameterisations(
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
    assert proposal.parameters == ["x"]


@patch("nessai.proposal.flowproposal.base.CombinedReparameterisation")
@pytest.mark.parametrize("parameters_value", ["y", ["y"], ("y",)])
def test_configure_reparameterisations_dict_w_params(
    mocked_class, proposal, dummy_rc, dummy_cmb_rc, parameters_value
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
    proposal.map_to_unit_hypercube = False
    proposal.prior_bounds = proposal.model.bounds

    with patch(
        "nessai.proposal.flowproposal.base.CombinedReparameterisation",
        return_value=dummy_cmb_rc,
    ) as mocked_class:
        BaseFlowProposal.configure_reparameterisations(
            proposal,
            {
                "x": {
                    "reparameterisation": "default",
                    "parameters": parameters_value,
                }
            },
        )

    proposal.get_reparameterisation.assert_called_once_with("default")
    proposal.add_default_reparameterisations.assert_not_called()
    dummy_rc.assert_called_once_with(
        prior_bounds={"x": [-1, 1], "y": [-1, 1]},
        parameters=["x", "y"],
    )
    mocked_class.assert_called_once()
    # fmt: off
    proposal._reparameterisation.add_reparameterisations \
        .assert_called_once_with("r")
    # fmt: on

    assert proposal.parameters == ["x", "y"]


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
        BaseFlowProposal.configure_reparameterisations(
            proposal, {"x": {"scale": 1.0}}
        )

    assert "No reparameterisation found for x" in str(excinfo.value)


def test_configure_reparameterisations_dict_str(proposal):
    """Test configuration for reparameterisations dictionary from a str"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]
    proposal.fallback_reparameterisation = None
    BaseFlowProposal.configure_reparameterisations(proposal, {"x": "default"})

    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.prime_parameters == ["x_prime", "y"]
    assert proposal._reparameterisation.parameters == ["x", "y"]
    assert proposal._reparameterisation.prime_parameters == ["x_prime", "y"]


def test_configure_reparameterisations_dict_reparam(proposal):
    """Test configuration for reparameterisations dictionary"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]
    proposal.fallback_reparameterisation = None
    BaseFlowProposal.configure_reparameterisations(
        proposal, {"default": {"parameters": ["x"]}}
    )

    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.prime_parameters == ["x_prime", "y"]
    assert proposal._reparameterisation.parameters == ["x", "y"]
    assert proposal._reparameterisation.prime_parameters == ["x_prime", "y"]


def test_configure_reparameterisations_dict_reparam_list(proposal):
    """Test configuration for reparameterisations dictionary"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]
    proposal.fallback_reparameterisation = None
    BaseFlowProposal.configure_reparameterisations(
        proposal, {"default": ["x", "y"]}
    )

    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.prime_parameters == ["x_prime", "y_prime"]
    assert proposal._reparameterisation.parameters == ["x", "y"]
    assert proposal._reparameterisation.prime_parameters == [
        "x_prime",
        "y_prime",
    ]


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
    BaseFlowProposal.configure_reparameterisations(
        proposal,
        {"z-score": {"parameters": parameters}},
    )

    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.prime_parameters == ["x_0_prime", "x_1_prime", "y"]
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
    BaseFlowProposal.configure_reparameterisations(proposal, None)
    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.prime_parameters == ["x", "y"]

    assert proposal._reparameterisation.parameters == ["x", "y"]
    assert proposal._reparameterisation.prime_parameters == ["x", "y"]
    assert all(
        [
            isinstance(r, NullReparameterisation)
            for r in proposal._reparameterisation.reparameterisations.values()
        ]
    )


def test_configure_reparameterisations_string(proposal):
    """Test configuration when input is a string"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]
    proposal.fallback_reparameterisation = None
    BaseFlowProposal.configure_reparameterisations(proposal, "rescaletobounds")
    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.prime_parameters == ["x_prime", "y_prime"]

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


def test_configure_reparameterisations_fallback(proposal):
    """Test configuration when input is None"""
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = get_reparameterisation
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]
    proposal.fallback_reparameterisation = "default"
    BaseFlowProposal.configure_reparameterisations(proposal, None)
    proposal.add_default_reparameterisations.assert_not_called()
    assert proposal.prime_parameters == ["x_prime", "y_prime"]

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
        BaseFlowProposal.configure_reparameterisations(proposal, ["default"])
    assert "must be a dictionary" in str(excinfo.value)


def test_configure_reparameterisations_incorrect_config_type(proposal):
    """Assert an error is raised when the config for a key is not a dictionary
    or a known reparameterisation.
    """
    proposal.model.names = ["x"]
    with pytest.raises(TypeError) as excinfo:
        BaseFlowProposal.configure_reparameterisations(proposal, {"x": ["a"]})
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
        BaseFlowProposal.configure_reparameterisations(proposal, reparam)
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
        BaseFlowProposal.configure_reparameterisations(
            proposal, {"default": {"update_bounds": True}}
        )
    assert "No parameters key" in str(excinfo.value)


def test_configure_reparameterisation_with_rng(proposal, rng):
    """Test that the rng is correctly passed to the reparameterisation if it is
    in the signature."""
    proposal.model.names = ["x"]
    proposal.model.bounds = {"x": [-1, 1]}
    proposal.get_reparameterisation = MagicMock(
        return_value=(
            Reparameterisation,
            {},
        )
    )
    proposal.rng = rng
    with patch("numpy.random.default_rng") as mocked_rng:
        BaseFlowProposal.configure_reparameterisations(
            proposal, {"default": {"parameters": ["x"]}}
        )
    mocked_rng.assert_not_called()
    assert proposal._reparameterisation["reparameterisation_x"].rng is rng


def test_set_parameter_order(proposal):
    proposal.model.names = ["x", "y"]
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.parameters = ["w", "x", "y", "z"]
    proposal._reparameterisation.values.return_value = [
        MagicMock(
            parameters=["x", "z"],
            prime_parameters=["x_prime", "z_prime"],
        ),
        MagicMock(
            parameters=["y"],
            prime_parameters=["y_prime", "y_aux"],
        ),
        # Derived parameter (isn't present in model)
        MagicMock(
            parameters=["w"],
            prime_parameters=["w_prime"],
        ),
    ]
    proposal._set_parameter_order()
    # Parameter order will be model + others in order returned by the
    # reparameterisation
    assert proposal.parameters == ["x", "y", "w", "z"]
    # Model parameters, additional parameters from reparams with model
    # parameters, then any remaining parameters from reparams in order
    assert proposal.prime_parameters == [
        "x_prime",
        "z_prime",
        "y_prime",
        "y_aux",
        "w_prime",
    ]


def test_set_rescaling_with_model(proposal, model):
    """
    Test setting the rescaling when the model contains reparmaeterisations.
    """
    proposal.model = model
    proposal.model.reparameterisations = {"x": "default"}
    proposal.expansion_fraction = None

    def update(self):
        proposal.parameters = model.names
        proposal.prime_parameters = ["x_prime"]

    proposal.configure_reparameterisations = MagicMock()
    proposal.configure_reparameterisations.side_effect = update

    BaseFlowProposal.set_rescaling(proposal)

    proposal.configure_reparameterisations.assert_called_once_with(
        {"x": "default"}
    )
    assert proposal.reparameterisations == {"x": "default"}
    assert proposal.prime_parameters == ["x_prime"]


def test_set_rescaling_with_reparameterisations(proposal, model):
    """
    Test setting the rescaling when a reparameterisations dict is defined.
    """
    proposal.model = model
    proposal.model.reparameterisations = None
    proposal.reparameterisations = {"x": "default"}
    proposal.expansion_fraction = None

    def update(self):
        proposal.parameters = model.names
        proposal.prime_parameters = ["x_prime"]

    proposal.configure_reparameterisations = MagicMock()
    proposal.configure_reparameterisations.side_effect = update

    BaseFlowProposal.set_rescaling(proposal)

    proposal.configure_reparameterisations.assert_called_once_with(
        {"x": "default"}
    )
    assert proposal.reparameterisations == {"x": "default"}
    assert proposal.prime_parameters == ["x_prime"]


@pytest.mark.parametrize("n", [1, 10])
def test_rescale(proposal, n, map_to_unit_hypercube):
    """Test rescaling when using reparameterisation dict"""
    x = numpy_array_to_live_points(np.random.randn(n, 2), ["x", "y"])
    x["logL"] = np.random.randn(n)
    x["logP"] = np.random.randn(n)
    x_prime = numpy_array_to_live_points(
        np.random.randn(n, 2), ["x_prime", "y_prime"]
    )
    proposal.x_prime_dtype = get_dtype(["x_prime", "y_prime"])
    proposal.map_to_unit_hypercube = map_to_unit_hypercube
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.reparameterise = MagicMock(
        return_value=[x, x_prime, np.ones(x.size)]
    )
    proposal.model.to_unit_hypercube = MagicMock(side_effect=lambda a: a)

    x_prime_out, log_j = BaseFlowProposal.rescale(
        proposal, x, compute_radius=False, test="lower"
    )

    np.testing.assert_array_equal(
        x_prime[["x_prime", "y_prime"]], x_prime_out[["x_prime", "y_prime"]]
    )
    np.testing.assert_array_equal(
        x[["logP", "logL"]], x_prime_out[["logP", "logL"]]
    )
    proposal._reparameterisation.reparameterise.assert_called_once()
    if map_to_unit_hypercube:
        proposal.model.to_unit_hypercube.assert_called_once_with(x)
    else:
        proposal.model.to_unit_hypercube.assert_not_called()


@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("return_unit_hypercube", [True, False])
def test_inverse_rescale(
    proposal, n, map_to_unit_hypercube, return_unit_hypercube
):
    """Test rescaling when using reparameterisation dict"""
    x = numpy_array_to_live_points(np.random.randn(n, 2), ["x", "y"]).squeeze()
    x_prime = numpy_array_to_live_points(
        np.random.randn(n, 2), ["x_prime", "y_prime"]
    )
    x_prime["logL"] = np.random.randn(n)
    x_prime["logP"] = np.random.randn(n)
    proposal.map_to_unit_hypercube = map_to_unit_hypercube
    proposal.x_dtype = get_dtype(["x", "y"])
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.inverse_reparameterise = MagicMock(
        return_value=[x, x_prime, np.ones(x.size)]
    )
    proposal.model.from_unit_hypercube = MagicMock(side_effect=lambda a: a)

    x_out, log_j = BaseFlowProposal.inverse_rescale(
        proposal, x_prime, return_unit_hypercube=return_unit_hypercube
    )

    np.testing.assert_array_equal(x[["x", "y"]], x_out[["x", "y"]])
    np.testing.assert_array_equal(
        x_prime[["logP", "logL"]], x_out[["logP", "logL"]]
    )
    proposal._reparameterisation.inverse_reparameterise.assert_called_once()
    if map_to_unit_hypercube and not return_unit_hypercube:
        proposal.model.from_unit_hypercube.assert_called_once_with(x)
    else:
        proposal.model.from_unit_hypercube.assert_not_called()


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
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.one_to_one = True

    BaseFlowProposal.verify_rescaling(proposal)

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
    proposal._reparameterisation.reset.assert_called_once()


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
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.one_to_one = True

    with pytest.raises(
        ReparameterisationError, match=r"Rescaling is not invertible .*"
    ):
        BaseFlowProposal.verify_rescaling(proposal)


def test_verify_rescaling_invertible_error_dynamic_range(proposal):
    """Assert an error is raised if the rescaling is not invertible"""
    x = np.array([[1e-15], [1e15]], dtype=[("x", "f8")])
    x_prime = x["x"] / 2
    log_j = np.array([-2, -2])
    x_out = x.copy()[::-1]
    log_j_inv = np.array([2, 2])

    proposal.model.new_point = MagicMock(return_value=x)
    proposal.rescale = MagicMock(return_value=(x_prime, log_j))
    proposal.inverse_rescale = MagicMock(return_value=(x_out, log_j_inv))
    proposal.rescaling_set = True
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.one_to_one = True

    with pytest.raises(ReparameterisationError, match=r"dynamic range"):
        BaseFlowProposal.verify_rescaling(proposal)


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
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.one_to_one = True

    with pytest.raises(RuntimeError) as excinfo:
        BaseFlowProposal.verify_rescaling(proposal)
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
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.one_to_one = True

    with pytest.raises(RuntimeError) as excinfo:
        BaseFlowProposal.verify_rescaling(proposal)
    assert "Rescaling Jacobian is not invertible" in str(excinfo.value)


def test_verify_rescaling_rescaling_not_set(proposal):
    """Assert an error is raised if the rescaling is not set"""
    proposal.rescaling_set = False
    with pytest.raises(RuntimeError, match=r"Rescaling must be set .*"):
        BaseFlowProposal.verify_rescaling(proposal)


def test_verify_rescaling_not_one_to_one(proposal, caplog):
    proposal.rescaling_set = True
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.one_to_one = False
    proposal.model.new_point = MagicMock()
    BaseFlowProposal.verify_rescaling(proposal)
    assert "Could not check if reparameterisation is invertible" in str(
        caplog.text
    )
    proposal.model.new_point.assert_not_called()


def test_check_state_update(proposal, map_to_unit_hypercube):
    """Assert the update method is called"""
    x = numpy_array_to_live_points(np.random.randn(10, 2), ["x", "y"])
    x_hyper = x.copy()
    proposal.map_to_unit_hypercube = map_to_unit_hypercube
    proposal._reparameterisation = Mock()
    proposal._reparameterisation.update = MagicMock()
    proposal.model.to_unit_hypercube = MagicMock(return_value=x_hyper)
    BaseFlowProposal.check_state(proposal, x)
    if map_to_unit_hypercube:
        proposal.model.to_unit_hypercube.assert_called_once_with(x)
        proposal._reparameterisation.update.assert_called_once_with(x_hyper)
    else:
        proposal._reparameterisation.update.assert_called_once_with(x)


def test_multi_parameter_reparameterisation_ordering(rng):
    """Parameter-key grouping should preserve model-order parameters."""
    grouped = make_test_proposal(
        {
            "a": {"reparameterisation": "null", "parameters": ["d"]},
            "b": "null",
            "c": "null",
        },
        rng=rng,
    )

    assert grouped.parameters == ["a", "b", "c", "d"]
    assert grouped.prime_parameters == ["a", "b", "c", "d"]


@pytest.mark.integration_test
def test_multi_parameter_reparameterisation_flow_input_order(rng):
    baseline = make_test_proposal(
        {"a": "null", "b": "null", "c": "null", "d": "null"},
        rng=rng,
    )
    grouped = make_test_proposal(
        {
            "a": {"reparameterisation": "null", "parameters": ["d"]},
            "b": "null",
            "c": "null",
        },
        rng=rng,
    )
    x = numpy_array_to_live_points(
        np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]),
        ["a", "b", "c", "d"],
    )

    x_prime_baseline, _ = baseline.rescale(x)
    x_prime_grouped, _ = grouped.rescale(x)

    np.testing.assert_array_equal(
        live_points_to_array(
            x_prime_baseline, names=baseline.prime_parameters, copy=True
        ),
        np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]),
    )
    np.testing.assert_array_equal(
        live_points_to_array(
            x_prime_grouped, names=grouped.prime_parameters, copy=True
        ),
        np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]),
    )

    baseline.flow = MagicMock()
    baseline.flow.forward_and_log_prob = MagicMock(
        return_value=(np.zeros((x.size, 4)), np.zeros(x.size))
    )
    grouped.flow = MagicMock()
    grouped.flow.forward_and_log_prob = MagicMock(
        return_value=(np.zeros((x.size, 4)), np.zeros(x.size))
    )

    baseline.forward_pass(x)
    grouped.forward_pass(x)

    np.testing.assert_array_equal(
        baseline.flow.forward_and_log_prob.call_args[0][0],
        np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]),
    )
    np.testing.assert_array_equal(
        grouped.flow.forward_and_log_prob.call_args[0][0],
        np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]),
    )


@pytest.mark.integration_test
def test_multi_parameter_reparameterisation_ordering_integration(rng):
    """Grouped scaling should affect only `a` and `d` in stable order."""
    baseline = make_test_proposal(
        {
            "a": {"reparameterisation": "rescale", "scale": 2.0},
            "b": "null",
            "c": "null",
            "d": {"reparameterisation": "rescale", "scale": 10.0},
        },
        rng=rng,
    )
    grouped = make_test_proposal(
        {
            "rescale": {
                "parameters": ["a", "d"],
                "scale": {"a": 2.0, "d": 10.0},
            },
            "b": "null",
            "c": "null",
        },
        rng=rng,
    )
    x = numpy_array_to_live_points(
        np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]),
        ["a", "b", "c", "d"],
    )

    x_prime_baseline, log_j_baseline = baseline.rescale(x)
    x_prime_grouped, log_j_grouped = grouped.rescale(x)
    x_out_grouped, log_j_inv_grouped = grouped.inverse_rescale(x_prime_grouped)

    expected = np.array([[0.5, 2.0, 3.0, 0.4], [5.0, 20.0, 30.0, 4.0]])
    expected_log_j = -np.log(20.0) * np.ones(x.shape[0])

    np.testing.assert_array_equal(
        live_points_to_array(
            x_prime_baseline, names=baseline.prime_parameters, copy=True
        ),
        expected,
    )
    np.testing.assert_array_equal(
        live_points_to_array(
            x_prime_grouped, names=grouped.prime_parameters, copy=True
        ),
        expected,
    )
    np.testing.assert_allclose(log_j_baseline, expected_log_j)
    np.testing.assert_allclose(log_j_grouped, expected_log_j)
    np.testing.assert_array_equal(
        live_points_to_array(x_out_grouped, names=grouped.model.names),
        live_points_to_array(x, names=grouped.model.names),
    )
    np.testing.assert_allclose(log_j_inv_grouped, -expected_log_j)

    baseline.flow = MagicMock()
    baseline.flow.forward_and_log_prob = MagicMock(
        return_value=(np.zeros((x.size, 4)), np.zeros(x.size))
    )
    grouped.flow = MagicMock()
    grouped.flow.forward_and_log_prob = MagicMock(
        return_value=(np.zeros((x.size, 4)), np.zeros(x.size))
    )

    baseline.forward_pass(x)
    grouped.forward_pass(x)

    np.testing.assert_array_equal(
        baseline.flow.forward_and_log_prob.call_args[0][0], expected
    )
    np.testing.assert_array_equal(
        grouped.flow.forward_and_log_prob.call_args[0][0], expected
    )
