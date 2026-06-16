"""Test methods related to reparameterisations"""

from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from nessai.livepoint import (
    get_dtype,
    numpy_array_to_live_points,
)
from nessai.model import Model
from nessai.proposal.flowproposal.base import BaseFlowProposal
from nessai.reparameterisations import (
    Reparameterisation,
    ReparameterisationError,
)
from nessai.reparameterisations.utils import ReparameterisationSpec


@pytest.fixture
def proposal(proposal):
    """Specific mocked proposal for reparameterisation tests"""
    proposal.use_default_reparameterisations = False
    proposal.reverse_reparameterisations = False
    proposal.model = MagicMock(spec=Model)
    proposal.fallback_reparameterisation = None
    proposal.prior_bounds = {}
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
    m.add_reparameterisations = MagicMock()
    m.values.return_value = []
    m.check_order = MagicMock()
    m.parameters = []
    m.prime_parameters = []
    return m


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
def test_configure_reparameterisations(
    proposal,
    dummy_cmb_rc,
    reverse_order,
    use_default_reparameterisations,
):
    """Test configuration for reparameterisations dictionary.

    Also tests to make sure boundary inversion is set and if the
    `reverse_reparameterisation` is correctly set.
    """
    proposal.add_default_reparameterisations = MagicMock()
    proposal.map_to_unit_hypercube = False
    proposal.model = MagicMock()
    proposal.model.bounds = {"x": [-1, 1]}
    proposal.model.names = ["x"]
    proposal.fallback_reparameterisation = None
    proposal.reverse_reparameterisations = reverse_order
    proposal.use_default_reparameterisations = use_default_reparameterisations
    proposal.prior_bounds = proposal.model.bounds

    dummy_cmb_rc.parameters = ["x"]
    reparams = [MagicMock()]
    proposal.instantiate_reparameterisation_from_spec = MagicMock(
        side_effect=reparams
    )
    specs = [MagicMock()]
    proposal._set_parameter_order = MagicMock()

    reparameterisations = {"x": {"reparameterisation": "default"}}

    with (
        patch(
            "nessai.proposal.flowproposal.base.CombinedReparameterisation",
            return_value=dummy_cmb_rc,
        ) as mocked_class,
        patch(
            "nessai.proposal.flowproposal.base.parse_reparameterisations",
            return_value=specs,
        ) as mocked_parse,
    ):
        BaseFlowProposal.configure_reparameterisations(
            proposal, reparameterisations
        )

    mocked_parse.assert_called_once_with(
        reparameterisations,
        model_names=proposal.model.names,
        class_name=proposal.__class__.__name__,
    )

    proposal.instantiate_reparameterisation_from_spec.assert_has_calls(
        [call(spec) for spec in specs]
    )
    proposal._reparameterisation.add_reparameterisations.assert_has_calls(
        [call(r) for r in reparams]
    )

    if use_default_reparameterisations:
        proposal.add_default_reparameterisations.assert_called_once()
    else:
        proposal.add_default_reparameterisations.assert_not_called()

    # other_params should be empty
    proposal.get_reparameterisation.assert_not_called()

    mocked_class.assert_called_once_with(
        reverse_order=reverse_order, initial_parameters=["x"]
    )

    proposal._set_parameter_order.assert_called_once()


def test_get_reparameterisation_from_spec_model_key(proposal, dummy_rc):
    """Assert parameter-key specs are converted into a config."""
    proposal.get_reparameterisation = MagicMock(
        return_value=(dummy_rc, {"offset": 1.0})
    )
    spec = ReparameterisationSpec(
        source_key="x",
        spec_index=0,
        reparameterisation="default",
        source_is_parameter=True,
        input_parameters=["x", "y"],
        kwargs={"scale": 2.0},
    )

    rc, config = BaseFlowProposal.get_reparameterisation_from_spec(
        proposal, spec
    )

    assert rc is dummy_rc
    assert config == {
        "offset": 1.0,
        "input_parameters": ["x", "y"],
        "scale": 2.0,
    }
    proposal.get_reparameterisation.assert_called_once_with("default")


def test_get_prior_bounds_for_parameters_allows_missing_auxiliary(proposal):
    """Assert auxiliary parameters are ignored in the default lookup mode."""
    proposal.prior_bounds = {"x": [-1, 1]}
    proposal.model.bounds = {"x": [-10, 10], "aux": [0, 1]}

    out = BaseFlowProposal._get_prior_bounds_for_parameters(
        proposal, ["x", "aux"]
    )

    assert out == {"x": [-1, 1]}


def test_get_prior_bounds_for_parameters_single_parameter(proposal):
    proposal.prior_bounds = {"x": [-1, 1]}
    proposal.model.bounds = {"x": [-10, 10]}

    out = BaseFlowProposal._get_prior_bounds_for_parameters(proposal, "x")

    assert out == {"x": [-1, 1]}


def test_get_prior_bounds_for_parameters_unknown_parameter(proposal):
    proposal.prior_bounds = {"x": [-1, 1]}
    proposal.model.bounds = {"x": [-10, 10]}

    out = BaseFlowProposal._get_prior_bounds_for_parameters(proposal, "y")

    assert out is None


@pytest.mark.parametrize(
    "parameters",
    [
        "x.*",
        ["x.*"],
        ("x.*",),
    ],
)
def test_get_reparameterisation_from_spec_resolves_patterns(
    proposal, dummy_rc, parameters
):
    """Assert non-parameter specs resolve regex patterns."""
    proposal.get_reparameterisation = MagicMock(return_value=(dummy_rc, {}))
    proposal.model.names = ["x_0", "x_1", "y"]
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.parameters = ["x_0", "x_1", "y"]
    spec = ReparameterisationSpec(
        source_key="z-score",
        spec_index=0,
        reparameterisation="z-score",
        source_is_parameter=False,
        input_parameters=parameters,
        kwargs={},
    )

    rc, config = BaseFlowProposal.get_reparameterisation_from_spec(
        proposal, spec
    )

    assert rc is dummy_rc
    assert config == {"input_parameters": ["x_0", "x_1"]}


@pytest.mark.parametrize(
    "spec",
    [
        ReparameterisationSpec(
            source_key="x",
            spec_index=0,
            reparameterisation="sine",
            source_is_parameter=True,
            input_parameters=["x"],
            kwargs={},
        ),
        ReparameterisationSpec(
            source_key="sine",
            spec_index=0,
            reparameterisation="sine",
            source_is_parameter=False,
            input_parameters=["x"],
            kwargs={},
        ),
    ],
)
def test_get_reparameterisation_from_spec_unknown(proposal, spec):
    """Assert unknown reparameterisations raise a runtime error."""
    proposal.get_reparameterisation = MagicMock(side_effect=ValueError)

    with pytest.raises(RuntimeError) as excinfo:
        BaseFlowProposal.get_reparameterisation_from_spec(proposal, spec)

    assert "is not a parameter in the model or a known" in str(excinfo.value)


def test_get_reparameterisation_from_spec_no_parameters(proposal, dummy_rc):
    """Assert an error is raised if the resolved config has no inputs."""
    proposal.get_reparameterisation = MagicMock(return_value=(dummy_rc, {}))
    proposal.model.names = ["x"]
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.parameters = ["x"]
    spec = ReparameterisationSpec(
        source_key="default",
        spec_index=0,
        reparameterisation="default",
        source_is_parameter=False,
        input_parameters=None,
        kwargs={"update_bounds": True},
    )

    with pytest.raises(RuntimeError) as excinfo:
        BaseFlowProposal.get_reparameterisation_from_spec(proposal, spec)

    assert "No input_parameters key" in str(excinfo.value)


def test_get_reparameterisation_from_spec_empty_parameters(proposal, dummy_rc):
    proposal.get_reparameterisation = MagicMock(return_value=(dummy_rc, {}))
    spec = ReparameterisationSpec(
        source_key="x",
        spec_index=0,
        reparameterisation="default",
        source_is_parameter=True,
        input_parameters=[],
        kwargs={},
    )

    with pytest.raises(RuntimeError) as excinfo:
        BaseFlowProposal.get_reparameterisation_from_spec(proposal, spec)

    assert "No input_parameters key" in str(excinfo.value)


def test_get_reparameterisation_from_spec_allows_prime_inputs(
    proposal, dummy_rc
):
    """Assert chained specs can use prime-space input parameters."""
    proposal.get_reparameterisation = MagicMock(return_value=(dummy_rc, {}))
    proposal.model.names = ["x"]
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.parameters = ["x"]
    proposal._reparameterisation.prime_parameters = ["x_bounded"]
    spec = ReparameterisationSpec(
        source_key="prime-scale",
        spec_index=1,
        reparameterisation="prime-scale",
        source_is_parameter=False,
        input_parameters=["x_bounded"],
        kwargs={
            "output_parameters": ["x_scaled"],
        },
    )

    rc, config = BaseFlowProposal.get_reparameterisation_from_spec(
        proposal, spec
    )

    assert rc is dummy_rc
    assert config == {
        "input_parameters": ["x_bounded"],
        "output_parameters": ["x_scaled"],
    }


def test_instantiate_reparameterisation_from_spec_with_rng(proposal, rng):
    """Test that the rng is passed through when supported by the class."""
    spec = Mock()
    proposal.rng = rng
    proposal.get_reparameterisation_from_spec = MagicMock(
        return_value=(Reparameterisation, {"input_parameters": ["x"]})
    )
    proposal._get_prior_bounds_for_parameters = MagicMock(
        return_value={"x": [-1, 1]}
    )

    with patch("numpy.random.default_rng") as mocked_rng:
        reparameterisation = (
            BaseFlowProposal.instantiate_reparameterisation_from_spec(
                proposal, spec
            )
        )

    mocked_rng.assert_not_called()
    proposal.get_reparameterisation_from_spec.assert_called_once_with(spec)
    proposal._get_prior_bounds_for_parameters.assert_called_once_with(["x"])
    assert reparameterisation.rng is rng


def test_configure_reparameterisations_dict_missing(proposal, dummy_cmb_rc):
    """Assert parser errors propagate out of configure."""
    proposal.model.names = ["x", "y"]
    proposal._set_parameter_order = MagicMock()

    with (
        patch(
            "nessai.proposal.flowproposal.base.CombinedReparameterisation",
            return_value=dummy_cmb_rc,
        ),
        patch(
            "nessai.proposal.flowproposal.base.parse_reparameterisations",
            side_effect=RuntimeError("No reparameterisation found for x"),
        ),
    ):
        with pytest.raises(RuntimeError) as excinfo:
            BaseFlowProposal.configure_reparameterisations(
                proposal, {"x": {"scale": 1.0}}
            )

    assert "No reparameterisation found for x" in str(excinfo.value)


def test_configure_reparameterisations_fallback(
    proposal, dummy_rc, dummy_cmb_rc
):
    """Assert the fallback reparameterisation is added for unclaimed parameters."""
    dummy_rc.return_value = "r"
    proposal.add_default_reparameterisations = MagicMock()
    proposal.get_reparameterisation = MagicMock(return_value=(dummy_rc, {}))
    proposal.instantiate_reparameterisation_from_spec = MagicMock()
    proposal._get_prior_bounds_for_parameters = MagicMock(
        return_value={"x": [-1, 1], "y": [-1, 1]}
    )
    proposal._set_parameter_order = MagicMock()
    proposal.model.bounds = {"x": [-1, 1], "y": [-1, 1]}
    proposal.model.names = ["x", "y"]
    proposal.prior_bounds = proposal.model.bounds
    proposal.fallback_reparameterisation = "default"

    with (
        patch(
            "nessai.proposal.flowproposal.base.CombinedReparameterisation",
            return_value=dummy_cmb_rc,
        ),
        patch(
            "nessai.proposal.flowproposal.base.parse_reparameterisations",
            return_value=[],
        ),
    ):
        BaseFlowProposal.configure_reparameterisations(proposal, None)

    proposal.add_default_reparameterisations.assert_not_called()
    proposal.get_reparameterisation.assert_called_once_with("default")
    proposal._get_prior_bounds_for_parameters.assert_called_once_with(
        ["x", "y"],
    )
    dummy_rc.assert_called_once_with(
        input_parameters=["x", "y"],
        prior_bounds={"x": [-1, 1], "y": [-1, 1]},
    )
    proposal._reparameterisation.add_reparameterisations.assert_called_once_with(
        "r"
    )
    proposal._set_parameter_order.assert_called_once()


def test_configure_reparameterisations_incorrect_type(proposal):
    """Assert an error is raised when input is not a dictionary"""
    proposal.model.names = []
    with pytest.raises(TypeError) as excinfo:
        BaseFlowProposal.configure_reparameterisations(proposal, ["default"])
    assert "must be a dictionary" in str(excinfo.value)


def test_configure_reparameterisations_incorrect_config_type(proposal):
    """Assert an error is raised when the config for a key is not a dictionary
    or a known reparameterisation.
    """
    proposal.model.names = ["x"]
    with pytest.raises(TypeError) as excinfo:
        BaseFlowProposal.configure_reparameterisations(proposal, {"x": [1]})
    assert "Unknown config type" in str(excinfo.value)


def test_set_parameter_order(proposal):
    proposal.model.names = ["x", "y"]
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.parameters = ["w", "x", "y", "z"]
    proposal._reparameterisation.values.return_value = [
        MagicMock(
            parameters=["x", "z"],
            output_parameters=["x_prime", "z_prime"],
            x_prime_input_parameters=[],
            x_prime_persistent_parameters=[],
        ),
        MagicMock(
            parameters=["y"],
            output_parameters=["y_prime", "y_aux"],
            x_prime_input_parameters=[],
            x_prime_persistent_parameters=[],
        ),
        # Derived parameter (isn't present in model)
        MagicMock(
            parameters=["w"],
            output_parameters=["w_prime"],
            x_prime_input_parameters=[],
            x_prime_persistent_parameters=[],
        ),
    ]
    BaseFlowProposal._set_parameter_order(proposal)
    assert proposal.parameters == ["x", "y", "w", "z"]
    assert proposal.prime_parameters == [
        "x_prime",
        "z_prime",
        "y_prime",
        "y_aux",
        "w_prime",
    ]


@pytest.mark.parametrize(
    "persistent, expected",
    [
        ([], ["x_scaled"]),
        (["x_bounded"], ["x_bounded", "x_scaled"]),
    ],
)
def test_set_parameter_order_removes_non_persistent_intermediate_prime_inputs(
    proposal, persistent, expected
):
    proposal.model.names = ["x"]
    proposal._reparameterisation = MagicMock()
    proposal._reparameterisation.parameters = ["x"]
    proposal._reparameterisation.values.return_value = [
        MagicMock(
            input_parameters=["x"],
            output_parameters=["x_bounded"],
            x_prime_input_parameters=[],
            x_prime_persistent_parameters=[],
        ),
        MagicMock(
            input_parameters=["x_bounded"],
            output_parameters=["x_scaled"],
            x_prime_input_parameters=["x_bounded"],
            x_prime_persistent_parameters=persistent,
        ),
    ]

    BaseFlowProposal._set_parameter_order(proposal)

    assert proposal._prime_parameters_internal == ["x_bounded", "x_scaled"]
    assert proposal.prime_parameters == expected


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
        proposal._prime_parameters_internal = proposal.prime_parameters

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
        proposal._prime_parameters_internal = proposal.prime_parameters

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
    proposal.x_prime_internal_dtype = get_dtype(["x_prime", "y_prime"])
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
