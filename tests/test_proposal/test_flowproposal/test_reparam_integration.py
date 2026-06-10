# -*- coding: utf-8 -*-
"""
Integration tests for adding the default reparameterisations
"""

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest

from nessai.livepoint import live_points_to_array, numpy_array_to_live_points
from nessai.model import Model
from nessai.proposal.flowproposal import FlowProposal
from nessai.proposal.flowproposal.base import BaseFlowProposal
from nessai.reparameterisations import (
    NullReparameterisation,
    Reparameterisation,
    default_reparameterisations,
    get_reparameterisation,
)

# General reparameterisations that do not need extra parameters
general_reparameterisations = {
    k: v
    for k, v in default_reparameterisations.items()
    if k not in ["scale", "rescale", "angle-pair", "scaleandshift"]
}


class DummyFlowProposal(BaseFlowProposal):
    def populate(self, worst_point, n_samples=10000):
        raise NotImplementedError


class AddAuxiliaryReparameterisation(Reparameterisation):
    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        auxiliary_parameter="aux",
        rng=None,
    ):
        super().__init__(
            parameters=parameters, prior_bounds=prior_bounds, rng=rng
        )
        self.auxiliary_parameter = auxiliary_parameter
        self.auxiliary_parameters = [auxiliary_parameter]

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        if self.auxiliary_parameter not in x.dtype.names:
            x_out = np.empty(
                x.shape,
                dtype=x.dtype.descr + [(self.auxiliary_parameter, "f8")],
            )
            for name in x.dtype.names:
                x_out[name] = x[name]
            x = x_out
        x[self.auxiliary_parameter] = 2.0 * x[self.parameters[0]]
        x_prime[self.prime_parameters[0]] = x[self.parameters[0]]
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        x[self.parameters[0]] = x_prime[self.prime_parameters[0]]
        x[self.auxiliary_parameter] = 2.0 * x[self.parameters[0]]
        return x, x_prime, log_j


class RescaleAuxiliaryReparameterisation(Reparameterisation):
    def __init__(
        self, parameters=None, prior_bounds=None, scale=10.0, rng=None
    ):
        super().__init__(
            parameters=parameters, prior_bounds=prior_bounds, rng=rng
        )
        self.scale = scale

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        x_prime[self.prime_parameters[0]] = x[self.parameters[0]] / self.scale
        log_j -= np.log(self.scale)
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        x[self.parameters[0]] = x_prime[self.prime_parameters[0]] * self.scale
        log_j += np.log(self.scale)
        return x, x_prime, log_j


class PrimeProducerReparameterisation(Reparameterisation):
    def reparameterise(self, x, x_prime, log_j, **kwargs):
        x_prime[self.prime_parameters[0]] = x[self.parameters[0]] + 1.0
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        x[self.parameters[0]] = x_prime[self.prime_parameters[0]] - 1.0
        return x, x_prime, log_j


class PrimeScaleReparameterisation(Reparameterisation):
    def reparameterise(self, x, x_prime, log_j, **kwargs):
        x_prime[self.prime_parameters[0]] = (
            2.0 * x_prime[self.prime_requires[0]]
        )
        log_j -= np.log(2.0)
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        x_prime[self.prime_requires[0]] = (
            x_prime[self.prime_parameters[0]] / 2.0
        )
        log_j += np.log(2.0)
        return x, x_prime, log_j


def make_auxiliary_test_proposal(reparameterisations, rng):
    model = create_autospec(Model)
    model.names = ["a", "b"]
    model.bounds = {n: [-1.0, 1.0] for n in model.names}
    model.reparameterisations = None

    proposal = DummyFlowProposal(
        model,
        rng=rng,
        poolsize=10,
        reparameterisations=reparameterisations,
        fallback_reparameterisation=None,
    )

    def get_test_reparameterisation(name):
        mapping = {
            "null": (NullReparameterisation, {}),
            "add-aux": (AddAuxiliaryReparameterisation, {}),
            "aux-rescale": (RescaleAuxiliaryReparameterisation, {}),
        }
        return mapping[name]

    proposal.get_reparameterisation = MagicMock(
        side_effect=get_test_reparameterisation
    )
    proposal.set_rescaling()
    return proposal


def make_prime_chain_test_proposal(reparameterisations, rng):
    model = create_autospec(Model)
    model.names = ["x"]
    model.bounds = {"x": [-1.0, 1.0]}
    model.reparameterisations = None

    proposal = DummyFlowProposal(
        model,
        rng=rng,
        poolsize=10,
        reparameterisations=reparameterisations,
        fallback_reparameterisation=None,
    )

    def get_test_reparameterisation(name):
        mapping = {
            "prime-producer": (PrimeProducerReparameterisation, {}),
            "prime-scale": (PrimeScaleReparameterisation, {}),
        }
        return mapping[name]

    proposal.get_reparameterisation = MagicMock(
        side_effect=get_test_reparameterisation
    )
    proposal.set_rescaling()
    return proposal


def make_test_proposal(reparameterisations, rng):
    model = create_autospec(Model)
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


@pytest.fixture(params=general_reparameterisations.keys())
def reparameterisation(request):
    return request.param


@pytest.fixture
def model():
    m = create_autospec(Model)
    m.names = ["x"]
    m.bounds = {"x": [-1, 1]}
    m.reparameterisations = None
    return m


@pytest.mark.integration_test
def test_configure_reparameterisations(tmpdir, model, reparameterisation):
    """Test adding one of the default reparameterisations.

    Only tests reparameterisations that don't need extra parameters.
    """
    proposal = FlowProposal(
        model,
        output=str(tmpdir.mkdir("test")),
        poolsize=10,
        reparameterisations={"x": reparameterisation},
    )
    proposal.set_rescaling()
    assert proposal._reparameterisation is not None


@pytest.mark.integration_test
@pytest.mark.parametrize("reparameterisation", ["scale", "rescale"])
def test_configure_reparameterisation_scale(tmpdir, model, reparameterisation):
    """Test adding the `Rescale` reparameterisation"""
    proposal = FlowProposal(
        model,
        output=str(tmpdir.mkdir("test")),
        poolsize=10,
        reparameterisations={
            "x": {"reparameterisation": reparameterisation, "scale": 2.0}
        },
    )
    proposal.set_rescaling()
    assert proposal._reparameterisation is not None


@pytest.mark.integration_test
def test_configure_reparameterisation_angle_pair(tmpdir, model):
    """Test adding the `AnglePair` reparameterisation"""
    model.names.append("y")
    model.bounds = {"x": [0, 2 * np.pi], "y": [-np.pi / 2, np.pi / 2]}
    proposal = FlowProposal(
        model,
        output=str(tmpdir.mkdir("test")),
        poolsize=10,
        reparameterisations={
            "x": {"reparameterisation": "angle-pair", "parameters": ["y"]}
        },
    )
    proposal.set_rescaling()
    assert proposal._reparameterisation is not None


@pytest.mark.integration_test
def test_default_reparameterisations(caplog, tmpdir):
    """Assert that by default the reparameterisation is z-score"""
    from nessai.reparameterisations import ScaleAndShift

    caplog.set_level("INFO")
    model = MagicMock()
    model.names = ["x1", "x10", "x11"]
    model.bounds = {p: [-1, 1] for p in model.names}
    model.reparameterisations = None
    proposal = FlowProposal(
        model, poolsize=100, output=str(tmpdir.mkdir("test"))
    )
    # Mocked model so can't verify rescaling
    proposal.verify_rescaling = MagicMock()
    proposal.initialise()
    reparams = list(proposal._reparameterisation.values())
    assert len(reparams) == 1
    assert reparams[0].parameters == ["x1", "x10", "x11"]
    assert proposal.prime_parameters == ["x1_prime", "x10_prime", "x11_prime"]
    assert isinstance(reparams[0], ScaleAndShift)
    assert reparams[0].estimate_scale is True
    assert reparams[0].estimate_shift is True


@pytest.mark.integration_test
def test_configure_reparameterisations_with_auxiliary_parameter(rng):
    proposal = make_auxiliary_test_proposal(
        {
            "a": {
                "reparameterisation": "add-aux",
                "auxiliary_parameter": "aux",
            },
            "aux-rescale": {"parameters": ["aux"], "scale": 10.0},
            "b": "null",
        },
        rng=rng,
    )

    assert proposal.parameters == ["a", "b", "aux"]
    assert proposal.prime_parameters == ["a_prime", "b", "aux_prime"]

    x = numpy_array_to_live_points(
        np.array([[1.0, 2.0], [10.0, 20.0]]),
        ["a", "b"],
    )

    x_prime, log_j = proposal.rescale(x)
    x_out, log_j_inv = proposal.inverse_rescale(x_prime)

    expected_prime = np.array([[1.0, 2.0, 0.2], [10.0, 20.0, 2.0]])
    expected_x = np.array([[1.0, 2.0, 2.0], [10.0, 20.0, 20.0]])
    expected_log_j = -np.log(10.0) * np.ones(x.shape[0])

    np.testing.assert_array_equal(
        live_points_to_array(
            x_prime, names=proposal.prime_parameters, copy=True
        ),
        expected_prime,
    )
    np.testing.assert_array_equal(
        live_points_to_array(x_out, names=proposal.parameters, copy=True),
        expected_x,
    )
    np.testing.assert_allclose(log_j, expected_log_j)
    np.testing.assert_allclose(log_j_inv, -expected_log_j)


@pytest.mark.integration_test
def test_configure_reparameterisations_with_prime_spec_chain(rng):
    proposal = make_prime_chain_test_proposal(
        {
            "x": [
                {
                    "reparameterisation": "prime-producer",
                    "prime_parameters": ["x_bounded"],
                },
                {
                    "reparameterisation": "prime-scale",
                    "parameters": [],
                    "prime_requires": ["x_bounded"],
                    "prime_parameters": ["x_scaled"],
                },
            ]
        },
        rng=rng,
    )

    assert proposal.parameters == ["x"]
    assert proposal.prime_parameters == ["x_bounded", "x_scaled"]

    x = numpy_array_to_live_points(np.array([[1.0], [2.0]]), ["x"])

    x_prime, log_j = proposal.rescale(x)
    x_out, log_j_inv = proposal.inverse_rescale(x_prime)

    expected_prime = np.array([[2.0, 4.0], [3.0, 6.0]])
    expected_x = np.array([[1.0], [2.0]])

    np.testing.assert_array_equal(
        live_points_to_array(
            x_prime, names=proposal.prime_parameters, copy=True
        ),
        expected_prime,
    )
    np.testing.assert_array_equal(
        live_points_to_array(x_out, names=proposal.parameters, copy=True),
        expected_x,
    )
    np.testing.assert_allclose(log_j, -np.log(2.0) * np.ones(x.shape[0]))
    np.testing.assert_allclose(log_j_inv, -log_j)


@pytest.mark.integration_test
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
