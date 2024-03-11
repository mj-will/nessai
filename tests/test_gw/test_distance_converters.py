# -*- coding: utf-8 -*-
"""
Test the distance converter classes.
"""
import numpy as np
import pytest
from scipy import stats
from unittest.mock import MagicMock, Mock, call, create_autospec, patch

from nessai.gw.utils import (
    ComovingDistanceConverter,
    DistanceConverter,
    NullDistanceConverter,
    PowerLawConverter,
    get_distance_converter,
)


@pytest.fixture
def base_converter():
    return create_autospec(DistanceConverter)


@pytest.fixture
def null_converter():
    return create_autospec(NullDistanceConverter)


@pytest.fixture
def power_law_converter():
    return create_autospec(PowerLawConverter)


@pytest.fixture
def comoving_vol_converter():
    return create_autospec(ComovingDistanceConverter)


def test_has_conversion():
    """Assert the values for has_conversion are correct"""
    assert DistanceConverter.has_conversion is False
    assert NullDistanceConverter.has_conversion is False
    assert PowerLawConverter.has_conversion is True
    assert ComovingDistanceConverter.has_conversion is True


def test_has_jacobian():
    """Assert the values of has_jacobian are corrects"""
    assert DistanceConverter.has_jacobian is False
    assert NullDistanceConverter.has_jacobian is True
    assert PowerLawConverter.has_jacobian is True
    assert ComovingDistanceConverter.has_jacobian is False


def test_converter_error():
    """Assert an error is raised if the methods are not implemented"""
    with pytest.raises(TypeError) as excinfo:
        DistanceConverter()
    assert "abstract methods" in str(excinfo.value)


def test_converter_to_uniform_parameter_error(base_converter):
    with pytest.raises(NotImplementedError):
        DistanceConverter.to_uniform_parameter(base_converter, None)


def test_converter_from_uniform_parameter_error(base_converter):
    with pytest.raises(NotImplementedError):
        DistanceConverter.from_uniform_parameter(base_converter, None)


def test_null_converter_init(null_converter, caplog):
    """Test the null distance converter init"""
    NullDistanceConverter.__init__(null_converter, d_min=10)
    assert "Kwargs {'d_min': 10} will be ignored" in caplog.text


def test_null_converter_to_uniform(null_converter):
    """Test the null distance converter to uniform"""
    d = np.arange(10)
    d_out, log_j = NullDistanceConverter.to_uniform_parameter(
        null_converter, d
    )
    np.testing.assert_array_equal(d_out, d)
    np.testing.assert_equal(log_j, 0)


def test_null_converter_from_uniform(null_converter):
    """Test the null distance converter from uniform"""
    d = np.arange(10)
    d_out, log_j = NullDistanceConverter.from_uniform_parameter(
        null_converter, d
    )
    np.testing.assert_array_equal(d_out, d)
    np.testing.assert_equal(log_j, 0)


def test_power_law_converter_missing_power():
    """Assert an error is raised if the power is not specified."""

    with pytest.raises(RuntimeError) as excinfo:
        PowerLawConverter(power=None)
    assert "Must specify the power" in str(excinfo.value)


@pytest.mark.parametrize("power, func", [(1, np.sqrt), (2, np.cbrt)])
def test_power_law_converter_init(power_law_converter, power, func):
    """Test the init method with powers of 1 or 2"""
    PowerLawConverter.__init__(power_law_converter, power=power, scale=200)
    assert power_law_converter._power == (power + 1)
    assert power_law_converter._f is func
    assert power_law_converter.scale == 200


@pytest.mark.parametrize("power", [3, 4, 5])
def test_power_law_converter_init_power_greater_than_2(
    power_law_converter, power
):
    """Test the init method for powers > 2"""
    x = 3
    PowerLawConverter.__init__(power_law_converter, power=power, scale=200)
    assert power_law_converter._power == (power + 1)
    assert power_law_converter._f(x) == (x ** (1 / (power + 1)))


@pytest.mark.parametrize(
    "d, scale, power, expected",
    [(1.0, 1.0, 3, np.log(4)), (1000.0, 1000.0, 2, -5.80914299)],
)
def test_power_law_jacobian(power_law_converter, d, scale, power, expected):
    """Test the power law Jacobian"""
    power_law_converter._power = power + 1
    power_law_converter.scale = scale
    log_j = PowerLawConverter._log_jacobian(power_law_converter, d)
    np.testing.assert_almost_equal(expected, log_j)


@pytest.mark.parametrize(
    "d, scale, power, expected",
    [(1.0, 1.0, 3, -np.log(4)), (1.0, 1000.0, 2, 5.80914299)],
)
def test_power_law_jacobian_inverse(
    power_law_converter, d, scale, power, expected
):
    """Test the power law Jacobian"""
    power_law_converter._power = power + 1
    power_law_converter.scale = scale
    log_j = PowerLawConverter._log_jacobian_inv(power_law_converter, d)
    np.testing.assert_almost_equal(expected, log_j)


def test_power_law_to_uniform_parameter(power_law_converter):
    """Test the conversion to a uniform parameter"""
    d = 20
    power_law_converter.scale = 10
    power_law_converter._power = 2
    power_law_converter._log_jacobian = Mock(return_value=2)
    du, lj = PowerLawConverter.to_uniform_parameter(power_law_converter, d)
    power_law_converter._log_jacobian.assert_called_once_with(d)
    assert du == 4
    assert lj == 2


def test_power_law_from_uniform_parameter(power_law_converter):
    """Test the conversion from a uniform parameter"""
    d = 4
    power_law_converter.scale = 10
    power_law_converter._f = Mock(return_value=2)
    power_law_converter._log_jacobian_inv = Mock(return_value=-2)
    du, lj = PowerLawConverter.from_uniform_parameter(power_law_converter, d)
    power_law_converter._f.assert_called_once_with(d)
    power_law_converter._log_jacobian_inv.assert_called_once_with(d)
    assert du == 20
    assert lj == -2


@pytest.mark.requires("astropy")
def test_comoving_vol_init(comoving_vol_converter):
    """Test the init method"""
    d_min = 100
    d_max = 1000
    dc_min = Mock()
    dc_min.value = 2
    dc_max = Mock()
    dc_max.value = 2000

    n_interp = 10
    dl_array = Mock()
    dl_array.value = np.arange(n_interp)
    dc_array = np.arange(n_interp)
    z_values = np.arange(n_interp + 2)

    # Patch hell - There has to be a better way
    with patch(
        "nessai.gw.utils.cosmo.Planck15", return_value=MagicMock()
    ), patch(
        "nessai.gw.utils.cosmo.Planck15.comoving_distance",
        side_effect=[dc_min, dc_max],
    ) as cd_mock, patch(
        "nessai.gw.utils.cosmo.Planck15.luminosity_distance",
        side_effect=[dl_array],
    ) as ld_mock, patch(
        "nessai.gw.utils.cosmo.z_at_value", side_effect=z_values
    ), patch(
        "numpy.linspace", return_value=dc_array
    ) as linspace_mock, patch(
        "nessai.gw.utils.interpolate.splrep", side_effect=["dc2dl", "dl2dc"]
    ) as interp_mock:
        ComovingDistanceConverter.__init__(
            comoving_vol_converter,
            d_min=d_min,
            d_max=d_max,
            n_interp=n_interp,
        )
    cd_mock.assert_has_calls([call(0), call(1)])
    ld_mock.assert_has_calls([call(z_values[2:].tolist())])

    assert comoving_vol_converter.dl_min == 95
    assert comoving_vol_converter.dl_max == 1050
    assert comoving_vol_converter.dc_min == 2
    assert comoving_vol_converter.dc_max == 2000
    linspace_mock.assert_called_once_with(2, 2000, 10)

    interp_mock.assert_has_calls(
        [call(dc_array, dl_array.value), call(dl_array.value, dc_array)]
    )
    assert comoving_vol_converter.interp_dc2dl == "dc2dl"
    assert comoving_vol_converter.interp_dl2dc == "dl2dc"


@pytest.mark.requires("astropy")
def test_comoving_vol_init_invalid_cosmology(comoving_vol_converter):
    """Test the init method when an invalid cosmology is given"""
    with pytest.raises(RuntimeError) as excinfo:
        ComovingDistanceConverter.__init__(
            comoving_vol_converter, cosmology="Planck"
        )
    assert "Could not get specified cosmology" in str(excinfo.value)


def test_comoving_vol_to_uniform(comoving_vol_converter):
    """Test converting to a uniform parameter"""
    d = 1000
    grid = Mock()
    comoving_vol_converter.interp_dl2dc = grid
    comoving_vol_converter.scale = 100
    with patch(
        "nessai.gw.utils.interpolate.splev", return_value=200
    ) as interp_mock:
        du, lj = ComovingDistanceConverter.to_uniform_parameter(
            comoving_vol_converter, d
        )
    interp_mock.assert_called_once_with(d, grid, ext=3)
    assert du == 8.0
    assert lj == 0


def test_comoving_vol_from_uniform(comoving_vol_converter):
    """Test converting from a uniform parameter"""
    d = 1000
    grid = Mock()
    comoving_vol_converter.interp_dc2dl = grid
    comoving_vol_converter.scale = 5
    with patch(
        "nessai.gw.utils.interpolate.splev", return_value=20
    ) as interp_mock:
        du, lj = ComovingDistanceConverter.from_uniform_parameter(
            comoving_vol_converter, d
        )
    # 5 * cbrt(1000) = 50
    interp_mock.assert_called_once_with(50, grid, ext=3)
    assert du == 20
    assert lj == 0


@pytest.mark.parametrize(
    "prior, cls",
    [
        ("uniform-comoving-volume", ComovingDistanceConverter),
        ("power-law", PowerLawConverter),
        ("other", NullDistanceConverter),
        (None, NullDistanceConverter),
    ],
)
def test_get_distance_converter(prior, cls):
    """Assert the correct class is returned."""
    cls_out = get_distance_converter(prior)
    assert cls_out == cls


@pytest.mark.parametrize("power", [1, 2, 3, 4])
@pytest.mark.flaky(reruns=5)
@pytest.mark.integration_test
def test_power_law_converter_distribution(power):
    """
    Check that the distribution of resulting samples is uniform when
    converting from a power law.
    """
    c = PowerLawConverter(power, scale=1)
    x = stats.powerlaw(power + 1).rvs(size=10000)
    y, _ = c.to_uniform_parameter(x)
    d, p = stats.kstest(y, "uniform")
    assert p >= 0.05


@pytest.mark.parametrize("power", [1, 2, 3, 4])
@pytest.mark.integration_test
def test_power_law_converter_inversion(power):
    """
    Check that the power law inversion is invertible
    """
    c = PowerLawConverter(power, scale=1)
    x = stats.powerlaw(power + 1).rvs(size=1000)
    y, _ = c.to_uniform_parameter(x)
    x_out, _ = c.from_uniform_parameter(y)
    np.testing.assert_array_almost_equal(x, x_out)


@pytest.mark.requires("astropy")
@pytest.mark.parametrize("cosmology", ["Planck15", "WMAP7"])
@pytest.mark.integration_test
def test_comoving_distance_converter_integration(cosmology):
    """Integration test for the comoving distance prior converter."""
    cdc = ComovingDistanceConverter(
        d_min=10, d_max=1000, cosmology=cosmology, scale=100, n_interp=10
    )
    u, lju = cdc.to_uniform_parameter(500)
    dl, ljd = cdc.from_uniform_parameter(500)
    assert u is not None
    assert dl is not None
    assert lju == 0
    assert ljd == 0


@pytest.mark.requires("astropy")
@pytest.mark.parametrize("cosmology", ["Planck15", "WMAP7"])
@pytest.mark.parametrize("scale", [1, 1000])
@pytest.mark.parametrize("n_interp", [200, 500])
@pytest.mark.integration_test
def test_comoving_distance_converter_inversion(cosmology, scale, n_interp):
    """Integration test to verify the conversion is invertible."""
    cdc = ComovingDistanceConverter(
        d_min=100,
        d_max=5000,
        cosmology=cosmology,
        scale=scale,
        n_interp=n_interp,
    )
    dl = np.random.uniform(100, 5000, 100)
    u, _ = cdc.to_uniform_parameter(dl)
    dl_out, _ = cdc.from_uniform_parameter(u)
    np.testing.assert_array_almost_equal(dl, dl_out)
