import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

import nessai.gw.utils as utils

from conftest import requires_dependency


@pytest.mark.parametrize("r, s, zero", [((0, np.pi), 2, 'bound'),
                                        ((0, 2 * np.pi), 1, 'bound'),
                                        ((0, 1), np.pi, 'bound'),
                                        ((-np.pi, np.pi), 1, 'centre')])
@pytest.mark.parametrize('radial', [False, True])
def test_angle_to_cartesian_to_angle(r, s, zero, radial):
    """
    Test the conversion from angle to cartesian and back for a range
    and scale.
    """
    n = 1000
    t = np.random.uniform(r[0], r[1], n)
    if radial:
        radii = np.random.rand(n)
    else:
        radii = None
    cart = utils.angle_to_cartesian(t, r=radii, scale=s)
    t_out = utils.cartesian_to_angle(*cart[:2], scale=s, zero=zero)
    assert_allclose(t, t_out[0])
    if radial:
        assert_allclose(radii, t_out[1])
    assert_allclose(cart[-1], -t_out[-1])


@pytest.mark.parametrize("r, s, zero", [((0, np.pi), 2, 'bound'),
                                        ((0, 2 * np.pi), 1, 'bound'),
                                        ((0, 1), np.pi, 'bound'),
                                        ((-np.pi, np.pi), 1, 'centre')])
def test_cartesian_to_angle_to_cartesian(r, s, zero):
    """
    Test the conversion from cartesian to angle and back.
    """
    cart = np.random.randn(2, 1000)
    angles = utils.cartesian_to_angle(*cart, scale=s, zero=zero)
    cart_out = utils.angle_to_cartesian(angles[0], r=angles[1], scale=s)

    assert_allclose(cart, cart_out[:2])
    assert_allclose(angles[-1], -cart_out[-1])


@pytest.mark.parametrize("dl", [True, False])
def test_ra_dec_to_cartesian_to_ra_dec(dl):
    """
    Test the conversion from ra, dec and dl to x, y, z and back. If dl=True
    the radial component is included.
    """
    sky = np.random.uniform((0, -np.pi / 2, 100),
                            (2 * np.pi, np.pi / 2, 10000), [1000, 3]).T
    if dl:
        cart = utils.ra_dec_to_cartesian(*sky)
    else:
        cart = utils.ra_dec_to_cartesian(*sky[:2], dL=None)

    sky_out = utils.cartesian_to_ra_dec(*cart[:3])
    assert_allclose(sky[:2], sky_out[:2])
    if dl:
        assert_allclose(sky[2], sky_out[2])
    assert_allclose(cart[3], -sky_out[3])


def test_cartesian_to_ra_dec_to_cartesian():
    """
    Assert that the coversion from x, y, z, to ra, dec, dL and back
    is self consistent
    """
    cart = np.random.randn(3, 1000)
    sky = utils.cartesian_to_ra_dec(*cart)
    cart_out = utils.ra_dec_to_cartesian(*sky[:3])
    assert_allclose(cart, cart_out[:3])
    assert_allclose(sky[3], -cart_out[3])


@pytest.mark.parametrize("dl", [True, False])
def test_az_zen_to_cartesian_to_az_zen(dl):
    """
    Test the conversion from azimuth, zenith and dl to x, y, z and back.
    If dl=True the radial component if included.
    """
    sky = np.random.uniform((0, 0, 100),
                            (2 * np.pi, np.pi, 10000), [1000, 3]).T
    if dl:
        cart = utils.azimuth_zenith_to_cartesian(*sky)
    else:
        cart = utils.azimuth_zenith_to_cartesian(*sky[:2], dL=None)

    sky_out = utils.cartesian_to_azimuth_zenith(*cart[:3])
    assert_allclose(sky[:2], sky_out[:2])
    if dl:
        assert_allclose(sky[2], sky_out[2])
    assert_allclose(cart[3], -sky_out[3])


def test_cartesian_to_az_zen_to_cartesian():
    """
    Assert that the coversion from x, y, z, to azimuth, zenith, dL and back
    is self consistent
    """
    cart = np.random.randn(3, 1000)
    sky = utils.cartesian_to_azimuth_zenith(*cart)
    cart_out = utils.azimuth_zenith_to_cartesian(*sky[:3])
    assert_allclose(cart, cart_out[:3])
    assert_allclose(sky[3], -cart_out[3])


@pytest.mark.parametrize("mode", ['split', 'duplicate', 'half'])
def test_zero_one_to_cartesian(mode):
    """
    Test is correctly applied when mapping [0, 1] to
    cartesian coordinates with a given mode.
    """
    x = np.random.rand(1000)
    cart = utils.zero_one_to_cartesian(x, mode=mode)
    x_out = utils.cartesian_to_zero_one(cart[0], cart[1])

    if mode == 'duplicate':
        assert_allclose(x, x_out[0][:x.size])
        assert_allclose(x, x_out[0][x.size:])
        assert_equal([c.size for c in cart], [2 * x.size for _ in cart])
    else:
        assert_allclose(x, x_out[0])
        assert_equal([c.size for c in cart], [x.size for _ in cart])

    if mode == 'half':
        assert (cart[1] >= 0).all()
    else:
        assert ((cart[0] < 0) & (cart[1] < 0)).any()
        assert ((cart[0] > 0) & (cart[1] > 0)).any()
        assert ((cart[0] < 0) & (cart[1] > 0)).any()
        assert ((cart[0] > 0) & (cart[1] < 0)).any()


def test_zero_one_to_cartesain_incorrect_mode():
    """
    Test to ensure that an incorrect mode raises an error
    """
    x = np.random.rand(1000)
    with pytest.raises(RuntimeError) as excinfo:
        utils.zero_one_to_cartesian(x, mode='roar')
    assert 'Unknown mode' in str(excinfo.value)


def test_cartesian_to_zero_one():
    """
    Test to ensure values are mapped to [0, 1]
    """
    cart = np.random.randn(2, 1000)
    x, _, _ = utils.cartesian_to_zero_one(*cart)
    assert np.logical_and(x >= 0, x <= 1).all()


@requires_dependency('lal')
def test_precessing_parameters():
    """
    Test to ensure spin coversions are invertible
    """
    n = 1000
    theta_jn = np.arccos(np.random.uniform(-1, 1, n))
    phi_jl = np.random.uniform(0, 2 * np.pi, n)
    phi_12 = np.random.uniform(0, 2 * np.pi, n)
    theta_1 = np.arccos(np.random.uniform(-1, 1, n))
    theta_2 = np.arccos(np.random.uniform(-1, 1, n))
    a_1 = np.random.uniform(0, 0.99, n)
    a_2 = np.random.uniform(0, 0.99, n)
    m1 = 36.0
    m2 = 29.0
    phase = 0.0
    f_ref = 50.0

    array_in = (theta_jn, phi_jl, theta_1, theta_2, phi_12, a_1, a_2)

    array_inter = utils.transform_from_precessing_parameters(
        *array_in, m1, m2, f_ref, phase)

    array_out = utils.transform_to_precessing_parameters(
        *array_inter[:-1], m1, m2, f_ref, phase)

    np.testing.assert_array_almost_equal(array_in, array_out[:-1])
    np.testing.assert_array_almost_equal(array_inter[-1], -array_out[-1])
