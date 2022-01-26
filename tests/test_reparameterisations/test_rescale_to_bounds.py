# -*- coding: utf-8 -*-
"""
Test the RescaleToBound class.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, call, create_autospec

from nessai.reparameterisations import RescaleToBounds
from nessai.livepoint import get_dtype, numpy_array_to_live_points


@pytest.fixture
def reparam():
    return create_autospec(RescaleToBounds)


@pytest.fixture()
def reparameterisation(model):
    def _get_reparameterisation(kwargs):
        return RescaleToBounds(parameters=model.names,
                               prior_bounds=model.bounds,
                               **kwargs)
    return _get_reparameterisation


@pytest.fixture(scope='function')
def assert_invertibility(model, n=100):
    def test_invertibility(reparam):
        x = model.new_point(N=n)
        x_prime = np.zeros([n], dtype=get_dtype(reparam.prime_parameters))
        log_j = 0

        assert x.size == x_prime.size

        x_re, x_prime_re, log_j_re = reparam.reparameterise(
            x, x_prime, log_j)

        np.testing.assert_array_equal(x, x_re)

        x_in = np.zeros([n], dtype=get_dtype(reparam.parameters))

        x_inv, x_prime_inv, log_j_inv = \
            reparam.inverse_reparameterise(x_in, x_prime_re, log_j)

        np.testing.assert_array_equal(x, x_inv)
        np.testing.assert_array_equal(x_prime_re, x_prime_inv)
        np.testing.assert_array_equal(log_j_re, -log_j_inv)

        return True

    return test_invertibility


@pytest.mark.parametrize(
    'rescale_bounds',
    [None, [0, 1], {'x': [0, 1], 'y': [-1, 1]}]
)
def test_rescale_bounds(reparameterisation, assert_invertibility,
                        rescale_bounds):
    """Test the different options for rescale to bounds"""
    reparam = reparameterisation({'rescale_bounds': rescale_bounds})
    if rescale_bounds is None:
        rescale_bounds = {p: [-1, 1] for p in reparam.parameters}
    elif isinstance(rescale_bounds, list):
        rescale_bounds = {p: rescale_bounds for p in reparam.parameters}

    assert reparam.rescale_bounds == rescale_bounds
    assert assert_invertibility(reparam)


def test_rescale_bounds_dict_missing_params(reparam):
    """Assert an error is raised if the rescale_bounds dict is missing a
    parameter.
    """
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.__init__(
            reparam,
            parameters=['x', 'y'],
            prior_bounds={'x': [-1, 1], 'y': [0, 1]},
            rescale_bounds={'x': [0, 1]}
        )
    assert 'Missing rescale bounds for parameters' in str(excinfo.value)


def test_rescale_bounds_incorrect_type(reparam):
    """Assert an error is raised if the rescale_bounds is an invalid type."""
    with pytest.raises(TypeError) as excinfo:
        RescaleToBounds.__init__(
            reparam,
            parameters=['x', 'y'],
            prior_bounds={'x': [-1, 1], 'y': [0, 1]},
            rescale_bounds=1,
        )
    assert 'must be an instance of list or dict' in str(excinfo.value)


@pytest.mark.parametrize(
    'boundary_inversion',
    [False, True, ['x'], {'x': 'split'}, {'x': 'inversion'}]
)
def test_boundary_inversion(reparameterisation, assert_invertibility,
                            boundary_inversion):
    """Test the different options for rescale to bounds"""
    reparam = reparameterisation({'boundary_inversion': boundary_inversion})

    assert assert_invertibility(reparam)


def test_boundary_inversion_invalid_type(reparam):
    """Assert an error is raised in the type is invalid"""
    with pytest.raises(TypeError) as excinfo:
        RescaleToBounds.__init__(
            reparam,
            parameters='x',
            prior_bounds=[0, 1],
            boundary_inversion='Yes',
        )
    assert 'boundary_inversion must be a list, dict or bool' \
        in str(excinfo.value)


def test_detect_edges_without_inversion(reparam):
    """Assert detect edges cannot be used with boundary inversion"""
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.__init__(
            reparam,
            parameters=['x', 'y'],
            prior_bounds={'x': [-1, 1], 'y': [0, 1]},
            detect_edges=True,
        )
    assert 'Must enable boundary inversion to use detect edges' \
        in str(excinfo.value)


def test_set_bounds(reparam):
    """Test the set bounds method."""
    reparam.parameters = ['x']
    reparam.rescale_bounds = {'x': np.array([-1, 1])}
    reparam.pre_rescaling = lambda x: (x / 2, np.zeros_like(x))
    reparam.offsets = {'x': 1}
    RescaleToBounds.set_bounds(reparam, {'x': np.array([-10, 10])})
    np.testing.assert_array_equal(reparam.pre_prior_bounds['x'], [-5, 5])
    np.testing.assert_array_equal(reparam.bounds['x'], [-6, 4])


def test_set_offets(reparam):
    """Assert the offset are set correctly"""
    reparam.pre_rescaling = lambda x: (x / 2, 0.0)

    RescaleToBounds.__init__(
        reparam,
        parameters=['x', 'y'],
        prior_bounds={'x': [8, 32], 'y': [2, 4]},
        offset=True,
    )

    assert reparam.offsets == {'x': 10.0, 'y': 1.5}


def test_reset_inversion(reparam):
    """Assert the edges are reset correctly"""
    reparam.parameters = ['x', 'y']
    reparam._edges = {'x': [-10, 10], 'y': [-5, 5]}
    RescaleToBounds.reset_inversion(reparam)
    assert reparam._edges == {'x': None, 'y': None}


def test_x_prime_log_prior_error(reparam):
    """Assert an error is raised if the prime prior is not defined."""
    reparam.has_prime_prior = False
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.x_prime_log_prior(reparam, 0.1)
    assert 'Prime prior is not configured' in str(excinfo.value)


def test_default_pre_rescaling(reparam):
    """Assert the default pre-rescaling is the identity"""
    x = np.array([1, 2, 3])
    expected_log_j = np.zeros(3)
    x_out, log_j = RescaleToBounds.pre_rescaling(reparam, x)
    x_out_inv, log_j_inv = RescaleToBounds.pre_rescaling_inv(reparam, x)

    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(x_out_inv, x)
    np.testing.assert_array_equal(log_j, expected_log_j)
    np.testing.assert_array_equal(log_j_inv, expected_log_j)


def test_default_post_rescaling(reparam):
    """Assert the default post-rescaling is the identity"""
    x = np.array([1, 2, 3])
    expected_log_j = np.zeros(3)
    x_out, log_j = RescaleToBounds.post_rescaling(reparam, x)
    x_out_inv, log_j_inv = RescaleToBounds.post_rescaling_inv(reparam, x)

    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(x_out_inv, x)
    np.testing.assert_array_equal(log_j, expected_log_j)
    np.testing.assert_array_equal(log_j_inv, expected_log_j)


def test_configure_pre_rescaling_none(reparam):
    """Test the configuration of the pre-rescaling if it is None"""
    RescaleToBounds.configure_pre_rescaling(reparam, None)
    assert reparam.has_pre_rescaling is False


def test_configure_post_rescaling_none(reparam):
    """Test the configuration of the post-rescaling if it is None"""
    RescaleToBounds.configure_post_rescaling(reparam, None)
    assert reparam.has_post_rescaling is False


def test_pre_rescaling_with_functions(reparam):
    """Assert that specifying functions works as intended"""
    rescaling = (np.exp, np.log)
    RescaleToBounds.configure_pre_rescaling(reparam, rescaling)
    assert reparam.has_pre_rescaling is True
    assert reparam.pre_rescaling is np.exp
    assert reparam.pre_rescaling_inv is np.log


def test_post_rescaling_with_functions(reparam):
    """Assert that specifying functions works as intended"""
    rescaling = (np.exp, np.log)
    RescaleToBounds.configure_post_rescaling(reparam, rescaling)
    assert reparam.has_post_rescaling is True
    assert reparam.has_prime_prior is False
    assert reparam.post_rescaling is np.exp
    assert reparam.post_rescaling_inv is np.log


def test_pre_rescaling_with_str(reparam):
    """Assert that specifying a str works as intended"""
    from nessai.utils.rescaling import rescaling_functions
    rescaling = 'logit'
    RescaleToBounds.configure_pre_rescaling(reparam, rescaling)
    assert reparam.has_pre_rescaling is True
    assert reparam.pre_rescaling is rescaling_functions['logit'][0]
    assert reparam.pre_rescaling_inv is rescaling_functions['logit'][1]


def test_pre_rescaling_with_invalid_str(reparam):
    """Assert an error is raised if the rescaling is not recognised"""
    rescaling = 'not_a_rescaling'
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.configure_pre_rescaling(reparam, rescaling)
    assert 'Unknown rescaling function: not_a_rescaling' in str(excinfo.value)


def test_post_rescaling_with_invalid_str(reparam):
    """Assert an error is raised if the rescaling is not recognised"""
    rescaling = 'not_a_rescaling'
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.configure_post_rescaling(reparam, rescaling)
    assert 'Unknown rescaling function: not_a_rescaling' in str(excinfo.value)


def test_post_rescaling_with_str(reparam):
    """Assert that specifying a str works as intended.

    Also test the config for the logit
    """
    reparam._update_bounds = False
    reparam.parameters = ['x']
    from nessai.utils.rescaling import rescaling_functions
    rescaling = 'logit'
    RescaleToBounds.configure_post_rescaling(reparam, rescaling)
    assert reparam.has_post_rescaling is True
    assert reparam.has_prime_prior is False
    assert reparam.post_rescaling is rescaling_functions['logit'][0]
    assert reparam.post_rescaling_inv is rescaling_functions['logit'][1]
    assert reparam.rescale_bounds == {'x': [0, 1]}


def test_post_rescaling_with_logit_update_bounds(reparam):
    """Assert an error is raised if using logit and update bounds"""
    reparam._update_bounds = True
    rescaling = 'logit'
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.configure_post_rescaling(reparam, rescaling)
    assert 'Cannot use logit with update bounds' in str(excinfo.value)


def test_pre_rescaling_invalid_input(reparam):
    """Assert an error is raised if the input isn't a str or tuple"""
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.configure_pre_rescaling(reparam, (np.exp, ))
    assert 'Pre-rescaling must be a str or tuple' in str(excinfo.value)


def test_post_rescaling_invalid_input(reparam):
    """Assert an error is raised if the input isn't a str or tuple"""
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.configure_post_rescaling(reparam, (np.exp, ))
    assert 'Post-rescaling must be a str or tuple' in str(excinfo.value)


def test_update_bounds_disabled(reparam, caplog):
    """Assert nothing happens in _update_bounds is False"""
    caplog.set_level('DEBUG')
    reparam._update_bounds = False
    RescaleToBounds.update_bounds(reparam, [0, 1])
    assert 'Update bounds not enabled' in str(caplog.text)


def test_update_bounds(reparam):
    """Assert the correct values are returned"""
    reparam.offsets = {'x': 0.0, 'y': 1.0}
    reparam.pre_rescaling = MagicMock(
        side_effect=lambda x: (x, np.zeros_like(x))
    )
    reparam.parameters = ['x', 'y']
    x = {'x': [-1, 0, 1], 'y': [-2, 0, 2]}
    RescaleToBounds.update_bounds(reparam, x)
    reparam.update_prime_prior_bounds.assert_called_once()
    reparam.pre_rescaling.assert_has_calls(
        [call(-1), call(1), call(-2), call(2)]
    )
    assert reparam.bounds == {'x': [-1, 1], 'y': [-3, 1]}


@pytest.mark.integration_test
def test_update_prime_prior_bounds_integration():
    """Assert the prime prior bounds are correctly computed"""
    rescaling = (
        lambda x: (x / 2, np.zeros_like(x)),
        lambda x: (2 * x, np.zeros_like(x)),
    )
    reparam = RescaleToBounds(
        parameters=['x'], prior_bounds=[1000, 1001], prior='uniform',
        pre_rescaling=rescaling, offset=True,
    )
    np.testing.assert_equal(reparam.offsets['x'], 500.25)
    np.testing.assert_array_equal(reparam.prior_bounds['x'], [1000, 1001])
    np.testing.assert_array_equal(reparam.pre_prior_bounds['x'], [500, 500.5])
    np.testing.assert_array_equal(reparam.bounds['x'], [-0.25, 0.25])
    np.testing.assert_array_equal(
        reparam.prime_prior_bounds['x_prime'], [-1, 1]
    )

    x_prime = numpy_array_to_live_points(
        np.array([[-2], [-1], [0.5], [1], [10]]), ['x_prime']
    )
    log_prior = reparam.x_prime_log_prior(x_prime)
    expected = np.array([-np.inf, 0, 0, 0, -np.inf])
    np.testing.assert_equal(log_prior, expected)
