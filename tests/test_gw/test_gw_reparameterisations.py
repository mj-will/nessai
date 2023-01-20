# -*- coding: utf-8 -*-
"""Tests for GW reparameterisations"""

import pytest
from nessai.gw.reparameterisations import (
    DeltaPhaseReparameterisation,
    DistanceReparameterisation,
    default_gw,
    get_gw_reparameterisation,
)
from nessai.utils.testing import assert_structured_arrays_equal
import numpy as np
from unittest.mock import MagicMock, patch, create_autospec

from nessai.livepoint import (
    dict_to_live_points,
    empty_structured_array,
)


@pytest.fixture
def distance_reparam():
    return create_autospec(DistanceReparameterisation)


@pytest.fixture
def delta_phase_reparam():
    return create_autospec(DeltaPhaseReparameterisation)


def test_get_gw_reparameterisation():
    """Test getting the gw reparameterisations.

    Assert the correct defaults are used.
    """
    expected = "out"
    with patch(
        "nessai.gw.reparameterisations.get_reparameterisation",
        return_value=expected,
    ) as base_fn:
        out = get_gw_reparameterisation("mass_ratio")
    assert out == expected
    base_fn.assert_called_once_with("mass_ratio", defaults=default_gw)


@pytest.mark.integration_test
def test_get_gw_reparameterisation_integration():
    """Integration test for get_gw_reparameterisation"""
    reparam, _ = get_gw_reparameterisation("distance")
    assert reparam is DistanceReparameterisation


@pytest.mark.parametrize(
    "has_conversion, has_jacobian, has_prime_prior, requires_prime_prior",
    [
        (False, True, False, False),
        (False, False, False, False),
        (True, False, True, True),
        (True, True, True, False),
    ],
)
def test_distance_reparameterisation_init(
    distance_reparam,
    has_conversion,
    has_jacobian,
    has_prime_prior,
    requires_prime_prior,
):
    """Test the init method for the DistanceReparameterisation class.

    Tests the different combinations of conversions and jacobians.
    """
    prior = "uniform-comoving-volume"
    parameter = "parameter"
    prior_bounds = {"parameter": [10.0, 100.0]}

    distance_reparam.detect_edges_kwargs = {}
    distance_reparam.requires_prime_prior = False
    distance_reparam.update_prime_prior_bounds = MagicMock()
    mock_converter = MagicMock()
    mock_converter.has_conversion = has_conversion
    mock_converter.has_jacobian = has_jacobian
    mock_converter_class = MagicMock(return_value=mock_converter)

    with patch(
        "nessai.gw.reparameterisations.get_distance_converter",
        return_value=mock_converter_class,
    ) as converter_fn:
        DistanceReparameterisation.__init__(
            distance_reparam,
            parameters=parameter,
            prior_bounds=prior_bounds,
            prior=prior,
        )

    converter_fn.assert_called_once_with(prior)
    mock_converter_class.assert_called_once_with(d_min=10.0, d_max=100.0)
    assert distance_reparam.has_prime_prior is has_prime_prior
    assert distance_reparam.requires_prime_prior is requires_prime_prior
    # If the reparam includes a conversion, the prime priors should be updated
    assert distance_reparam.update_prime_prior_bounds.called is has_conversion


def test_distance_reparameterisation_n_parameters_error(distance_reparam):
    """Assert an error is raised in more than one parameter is given."""
    prior = "uniform-comoving-volume"
    parameters = ["x", "y"]
    prior_bounds = {"x": [10.0, 100.0], "y": [10.0, 100.0]}
    with pytest.raises(RuntimeError) as excinfo:
        DistanceReparameterisation.__init__(
            distance_reparam,
            parameters=parameters,
            prior_bounds=prior_bounds,
            prior=prior,
        )

    assert "DistanceReparameterisation only supports one parameter" in str(
        excinfo.value
    )


def test_delta_phase_init(delta_phase_reparam):
    """Assert the parent method is called and the parameters are set."""
    parameters = "phase"
    prior_bounds = {"phase": [0, 6.28]}
    with patch(
        "nessai.gw.reparameterisations.Reparameterisation.__init__"
    ) as mock:
        DeltaPhaseReparameterisation.__init__(
            delta_phase_reparam,
            parameters=parameters,
            prior_bounds=prior_bounds,
        )
    mock.assert_called_once_with(
        parameters=parameters, prior_bounds=prior_bounds
    )
    assert delta_phase_reparam.requires == ["psi", "theta_jn"]
    assert delta_phase_reparam.prime_parameters == ["delta_phase"]


def test_delta_phase_reparameterise(delta_phase_reparam):
    """Assert the correct value is returned"""
    delta_phase_reparam.parameters = ["phase"]
    delta_phase_reparam.prime_parameters = ["delta_phase"]

    x = dict(phase=1.0, theta_jn=0.0, psi=0.5)
    x_prime = dict(delta_phase=np.nan, theta_jn=0.0, psi=0.5)
    log_j = 0

    (
        x_out,
        x_prime_out,
        log_j_out,
    ) = DeltaPhaseReparameterisation.reparameterise(
        delta_phase_reparam, x, x_prime, log_j
    )
    assert x_out == x
    assert x_prime_out["delta_phase"] == 0.5
    assert log_j_out == 0


def test_delta_phase_inverse_reparameterise(delta_phase_reparam):
    """Assert the correct value is returned"""
    delta_phase_reparam.parameters = ["phase"]
    delta_phase_reparam.prime_parameters = ["delta_phase"]

    x = dict(phase=np.nan, theta_jn=0.0, psi=0.5)
    x_prime = dict(delta_phase=0.5, theta_jn=0.0, psi=0.5)
    log_j = 0

    (
        x_out,
        x_prime_out,
        log_j_out,
    ) = DeltaPhaseReparameterisation.inverse_reparameterise(
        delta_phase_reparam, x, x_prime, log_j
    )
    assert x_prime_out == x_prime
    assert x["phase"] == 1.0
    assert log_j_out == 0


@pytest.mark.integration_test
def test_delta_phase_inverse_invertible():
    """Assert the reparameterisation is invertible"""
    n = 10
    parameters = ["phase"]
    prior_bounds = {"phase": [0.0, 2 * np.pi]}
    reparam = DeltaPhaseReparameterisation(
        parameters=parameters, prior_bounds=prior_bounds
    )
    x = dict_to_live_points(
        {
            "phase": np.random.uniform(0, 2 * np.pi, n),
            "psi": np.random.uniform(0, np.pi, n),
            "theta_jn": np.random.uniform(0, np.pi, n),
        }
    )
    x_prime = empty_structured_array(
        n, names=["delta_phase", "theta_jn", "psi"]
    )
    x_prime["psi"] = x["psi"]
    x_prime["theta_jn"] = x["theta_jn"]
    log_j = np.zeros(n)
    x_f, x_prime_f, log_j_f = reparam.reparameterise(x, x_prime, log_j)
    assert_structured_arrays_equal(x_f, x)
    np.testing.assert_array_equal(log_j_f, log_j)
    x_i, x_prime_i, log_j_i = reparam.reparameterise(x_f, x_prime_f, log_j_f)
    assert_structured_arrays_equal(x_prime_i, x_prime_f)
    assert_structured_arrays_equal(x_i, x)
    np.testing.assert_array_equal(log_j_i, log_j_f)
