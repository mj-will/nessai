import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

import nessai.gw.utils as utils


@pytest.mark.parametrize("r, s, zero", [((0, np.pi), 2, 'bound'),
                                        ((0, 2 * np.pi), 1, 'bound'),
                                        ((0, 1), np.pi, 'bound'),
                                        ((-np.pi, np.pi), 1, 'centre')])
def test_angle_to_cartesian_to_angle(r, s, zero):
    """
    Test the conversion from angle to cartesian and back for a range
    and scale.
    """
    t = np.random.uniform(r[0], r[1])
    cart = utils.angle_to_cartesian(t, scale=s)
    t_out = utils.cartesian_to_angle(*cart[:2], scale=s, zero=zero)
    assert_allclose(t, t_out[0])
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


@pytest.mark.parametrize("duplicate", [True, False])
def test_zero_one_to_cartesian_duplicate(duplicate):
    """
    Test duplicate is correctly applied when mapping [0, 1] to
    cartesian coordinates.
    """
    x = np.random.rand(1000)
    cart = utils.zero_one_to_cartesian(x, duplicate)
    if duplicate:
        assert_equal([c.size for c in cart], [2 * x.size for _ in cart])
    else:
        assert_equal([c.size for c in cart], [x.size for _ in cart])


def test_cartesian_to_zero_one():
    """
    Test to ensure values are mapped to [0, 1]
    """
    cart = np.random.randn(2, 1000)
    x, _, _ = utils.cartesian_to_zero_one(*cart)
    assert np.logical_and(x >= 0, x <= 1).all()
