# -*- coding: utf-8 -*-
"""Tests for GW reparameterisations"""

import pytest
from nessai.gw.reparameterisations import (
    DistanceReparameterisation,
    default_gw,
    get_gw_reparameterisation,
)
from unittest.mock import MagicMock, patch, create_autospec


@pytest.fixture
def distance_reparam():
    return create_autospec(DistanceReparameterisation)


def test_get_gw_reparameterisation():
    """Test getting the gw reparameterisations.

    Assert the correct defaults are used.
    """
    expected = 'out'
    with patch(
        'nessai.gw.reparameterisations.get_reparameterisation',
        return_value=expected,
    ) as base_fn:
        out = get_gw_reparameterisation('mass_ratio')
    assert out == expected
    base_fn.assert_called_once_with('mass_ratio', defaults=default_gw)


@pytest.mark.integration_test
def test_get_gw_reparameterisation_integration():
    """Integration test for get_gw_reparameterisation"""
    reparam, _ = get_gw_reparameterisation('distance')
    assert reparam is DistanceReparameterisation


@pytest.mark.parametrize(
    'has_conversion, has_jacobian, has_prime_prior, requires_prime_prior',
    [
        (False, True, False, False),
        (False, False, False, False),
        (True, False, True, True),
        (True, True, True, False),
    ]
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
    prior = 'uniform-comoving-volume'
    parameter = 'parameter'
    prior_bounds = {'parameter': [10.0, 100.0]}

    distance_reparam.detect_edges_kwargs = {}
    distance_reparam.requires_prime_prior = False
    distance_reparam.update_prime_prior_bounds = MagicMock()
    mock_converter = MagicMock()
    mock_converter.has_conversion = has_conversion
    mock_converter.has_jacobian = has_jacobian
    mock_converter_class = MagicMock(return_value=mock_converter)

    with patch(
        'nessai.gw.reparameterisations.get_distance_converter',
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
    prior = 'uniform-comoving-volume'
    parameters = ['x', 'y']
    prior_bounds = {'x': [10.0, 100.0], 'y': [10.0, 100.0]}
    with pytest.raises(RuntimeError) as excinfo:
        DistanceReparameterisation.__init__(
            distance_reparam,
            parameters=parameters,
            prior_bounds=prior_bounds,
            prior=prior,
        )

    assert 'DistanceReparameterisation only supports one parameter' \
        in str(excinfo.value)
