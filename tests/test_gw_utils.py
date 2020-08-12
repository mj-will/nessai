import numpy as np
from numpy.testing import assert_allclose
import flowproposal.gw.utils as utils


def test_sky_to_cartesian_to_sky():
    """
    Test the conversion from ra, dec and dl to x, y, z and back
    """
    sky = np.random.uniform((0, -np.pi / 2, 100),
                            (2 * np.pi, np.pi / 2, 10000), [1000, 3]).T
    cart = utils.sky_to_cartesian(*sky)
    sky_out = utils.cartesian_to_sky(*cart[:3])
    assert_allclose(sky, sky_out[:3])
    assert_allclose(cart[3], -sky_out[3])


def test_cartesian_to_sky_to_cartesian():
    """
    Assert that the coversion from x, y, z, to ra, dec, dL and back
    is self consistent
    """
    cart = np.random.randn(3, 1000)
    sky = utils.cartesian_to_sky(*cart)
    cart_out = utils.sky_to_cartesian(*sky[:3])
    assert_allclose(cart, cart_out[:3])
    assert_allclose(sky[3], -cart_out[3])
