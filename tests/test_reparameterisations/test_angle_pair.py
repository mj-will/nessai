# -*- coding: utf-8 -*-
"""
Test the AnglePair reparmeterisation
"""
import numpy as np
from numpy.testing import assert_equal
import pytest

from nessai.reparameterisations import AnglePair
from nessai.livepoint import get_dtype

angle_pairs = [(['ra', 'dec'], [[0, 2 * np.pi], [-np.pi / 2, np.pi / 2]]),
               (['az', 'zen'], [[0, 2 * np.pi], [0, np.pi]]),
               (['zen', 'az'], [[0, np.pi], [0, 2 * np.pi]])]


@pytest.fixture(params=angle_pairs, scope='function')
def angles(request):
    return request.param


@pytest.fixture(scope='function')
def assert_invertibility():
    def test_invertibility(reparam, angles, radial=None):

        n = list(angles.values())[0].size
        x = np.zeros([n], dtype=get_dtype(reparam.parameters))
        x_prime = np.zeros([n], dtype=get_dtype(reparam.prime_parameters))
        log_j = 0

        for a in reparam.angles:
            x[a] = angles[a]
        if radial is not None:
            x[reparam.radial] = radial

        assert x.size == x_prime.size

        x_re, x_prime_re, log_j_re = reparam.reparameterise(
            x, x_prime, log_j)

        for a in reparam.angles:
            np.testing.assert_array_equal(x[a], x_re[a])
        if radial is not None:
            np.testing.assert_array_equal(x[reparam.radial],
                                          x_re[reparam.radial])

        x_in = np.zeros([n], dtype=get_dtype(reparam.parameters))

        x_inv, x_prime_inv, log_j_inv = \
            reparam.inverse_reparameterise(x_in, x_prime_re, log_j)

        for a in reparam.angles:
            np.testing.assert_array_almost_equal(x[a], x_inv[a])
        if radial is not None:
            np.testing.assert_array_almost_equal(x[reparam.radial],
                                                 x_inv[reparam.radial])

        np.testing.assert_array_equal(x_prime_re, x_prime_inv)
        np.testing.assert_array_almost_equal(log_j_re, -log_j_inv)

        return True

    return test_invertibility


@pytest.mark.parametrize('parameters', ['x', ['w', 'x', 'y', 'z']])
def test_parameters_error(parameters):
    """
    Make sure reparameterisations fails with too many or too few parameters.
    """
    with pytest.raises(RuntimeError) as excinfo:
        AnglePair(parameters=parameters, prior_bounds=None)
    assert 'Must use a pair' in str(excinfo.value)


def test_two_angles(angles):
    """Test the reparmaterisation with just the angles"""
    parameters = angles[0]
    prior_bounds = {parameters[0]: angles[1][0],
                    parameters[1]: angles[1][1]}

    reparam = AnglePair(parameters=parameters, prior_bounds=prior_bounds)

    if 'ra' in parameters:
        assert reparam.convention == 'ra-dec'

    if 'az' in parameters:
        assert reparam.convention == 'az-zen'

    # Make sure parameter[0] is always ra or azimuth
    assert_equal(reparam.prior_bounds[reparam.angles[0]],
                 np.array([0, 2 * np.pi]))

    assert reparam.chi is not False
    assert hasattr(reparam.chi, 'rvs')
    assert reparam.has_prime_prior is False

    m = '_'.join(parameters[:2])
    assert reparam.angles == parameters[:2]
    assert reparam.radial == (m + '_radial')
    assert reparam.x == (m + '_x')
    assert reparam.y == (m + '_y')
    assert reparam.z == (m + '_z')


def test_ra_dec(assert_invertibility):
    """Test the invertibility when using RA and Dec"""
    parameters = ['ra', 'dec']
    prior_bounds = {parameters[0]: [0, 2 * np.pi],
                    parameters[1]: [-np.pi / 2, np.pi / 2]}

    reparam = AnglePair(parameters=parameters, prior_bounds=prior_bounds,
                        convention='ra-dec', prior='isotropic')

    n = 100
    angles = {'ra': np.random.uniform(*prior_bounds['ra'], n),
              'dec': np.arcsin(np.random.uniform(-1, 1, n))}
    assert assert_invertibility(reparam, angles)


def test_azimuth_zenith(assert_invertibility):
    """Test the inverbility when using azimuth and zenith"""
    parameters = ['az', 'zen']
    prior_bounds = {parameters[0]: [0, 2 * np.pi],
                    parameters[1]: [0, np.pi]}

    reparam = AnglePair(parameters=parameters, prior_bounds=prior_bounds,
                        convention='az-zen', prior='isotropic')

    n = 100
    angles = {'az': np.random.uniform(*prior_bounds['az'], n),
              'zen': np.arccos(np.random.uniform(-1, 1, n))}
    assert reparam.parameters[:2] == list(angles.keys())
    assert assert_invertibility(reparam, angles)


def test_w_radial(assert_invertibility):
    """Test the reparameterisation with a radial parameter"""
    parameters = ['r', 'ra', 'dec']
    prior_bounds = {parameters[0]: [0, 5],
                    parameters[1]: [0, 2 * np.pi],
                    parameters[2]: [-np.pi / 2, np.pi / 2]}

    reparam = AnglePair(parameters=parameters, prior_bounds=prior_bounds)

    assert reparam.parameters == ['ra', 'dec', 'r']
    assert reparam.angles == ['ra', 'dec']
    assert reparam.chi is False

    n = 100
    angles = {'ra': np.random.uniform(*prior_bounds['ra'], n),
              'dec': np.arcsin(np.random.uniform(-1, 1, n))}

    radial = np.random.uniform(*prior_bounds['r'], n)

    assert assert_invertibility(reparam, angles, radial=radial)
