# -*- coding: utf-8 -*-
"""
Test the AnglePair reparmeterisation
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, create_autospec

from nessai.reparameterisations import AnglePair
from nessai.livepoint import (
    get_dtype,
    numpy_array_to_live_points,
    parameters_to_live_point,
)

angle_pairs = [
    (["ra", "dec"], [[0, 2 * np.pi], [-np.pi / 2, np.pi / 2]]),
    (["dec", "ra"], [[0, 2 * np.pi], [-np.pi / 2, np.pi / 2]]),
    (["ra", "dec"], [[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]]),
    (["az", "zen"], [[0, 2 * np.pi], [0, np.pi]]),
    (["zen", "az"], [[0, np.pi], [0, 2 * np.pi]]),
]


@pytest.fixture(params=angle_pairs, scope="function")
def angles(request):
    return request.param


@pytest.fixture
def reparam():
    return create_autospec(AnglePair)


@pytest.fixture(scope="function")
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

        x_re, x_prime_re, log_j_re = reparam.reparameterise(x, x_prime, log_j)

        for a in reparam.angles:
            np.testing.assert_array_equal(x[a], x_re[a])
        if radial is not None:
            np.testing.assert_array_equal(
                x[reparam.radial], x_re[reparam.radial]
            )

        x_in = np.zeros([n], dtype=get_dtype(reparam.parameters))

        x_inv, x_prime_inv, log_j_inv = reparam.inverse_reparameterise(
            x_in, x_prime_re, log_j
        )

        for a in reparam.angles:
            np.testing.assert_array_almost_equal(x[a], x_inv[a])
        if radial is not None:
            np.testing.assert_array_almost_equal(
                x[reparam.radial], x_inv[reparam.radial]
            )

        np.testing.assert_array_equal(x_prime_re, x_prime_inv)
        np.testing.assert_array_almost_equal(log_j_re, -log_j_inv)

        return True

    return test_invertibility


@pytest.mark.parametrize("parameters", ["x", ["w", "x", "y", "z"]])
def test_parameters_error(parameters):
    """
    Make sure reparameterisations fails with too many or too few parameters.
    """
    with pytest.raises(RuntimeError) as excinfo:
        AnglePair(parameters=parameters, prior_bounds=None)
    assert "Must use a pair" in str(excinfo.value)


def test_two_angles(angles):
    """Test the reparmaterisation with just the angles"""
    parameters = angles[0]
    prior_bounds = {parameters[0]: angles[1][0], parameters[1]: angles[1][1]}

    reparam = AnglePair(parameters=parameters, prior_bounds=prior_bounds)

    if "ra" in parameters:
        assert reparam.convention == "ra-dec"

    if "az" in parameters:
        assert reparam.convention == "az-zen"

    # Make sure parameter[0] is always ra or azimuth
    assert np.ptp(reparam.prior_bounds[reparam.angles[0]]) == 2 * np.pi

    assert reparam.chi is not False
    assert hasattr(reparam.chi, "rvs")
    assert reparam.has_prime_prior is False

    m = "_".join(parameters[:2])
    assert reparam.angles == parameters[:2]
    assert reparam.radial == (m + "_radial")
    assert reparam.x == (m + "_x")
    assert reparam.y == (m + "_y")
    assert reparam.z == (m + "_z")


def test_ra_dec(assert_invertibility):
    """Test the invertibility when using RA and Dec"""
    parameters = ["ra", "dec"]
    prior_bounds = {
        parameters[0]: [0, 2 * np.pi],
        parameters[1]: [-np.pi / 2, np.pi / 2],
    }

    reparam = AnglePair(
        parameters=parameters,
        prior_bounds=prior_bounds,
        convention="ra-dec",
        prior="isotropic",
    )

    n = 100
    angles = {
        "ra": np.random.uniform(*prior_bounds["ra"], n),
        "dec": np.arcsin(np.random.uniform(-1, 1, n)),
    }
    assert assert_invertibility(reparam, angles)


def test_azimuth_zenith(assert_invertibility):
    """Test the inverbility when using azimuth and zenith"""
    parameters = ["az", "zen"]
    prior_bounds = {parameters[0]: [0, 2 * np.pi], parameters[1]: [0, np.pi]}

    reparam = AnglePair(
        parameters=parameters,
        prior_bounds=prior_bounds,
        convention="az-zen",
        prior="isotropic",
    )

    n = 100
    angles = {
        "az": np.random.uniform(*prior_bounds["az"], n),
        "zen": np.arccos(np.random.uniform(-1, 1, n)),
    }
    assert reparam.parameters[:2] == list(angles.keys())
    assert assert_invertibility(reparam, angles)


def test_w_radial(assert_invertibility):
    """Test the reparameterisation with a radial parameter"""
    parameters = ["r", "ra", "dec"]
    prior_bounds = {
        parameters[0]: [0, 5],
        parameters[1]: [0, 2 * np.pi],
        parameters[2]: [-np.pi / 2, np.pi / 2],
    }

    reparam = AnglePair(parameters=parameters, prior_bounds=prior_bounds)

    assert reparam.parameters == ["ra", "dec", "r"]
    assert reparam.angles == ["ra", "dec"]
    assert reparam.chi is False

    n = 100
    angles = {
        "ra": np.random.uniform(*prior_bounds["ra"], n),
        "dec": np.arcsin(np.random.uniform(-1, 1, n)),
    }

    radial = np.random.uniform(*prior_bounds["r"], n)

    assert assert_invertibility(reparam, angles, radial=radial)


@pytest.mark.parametrize(
    "convention, input, expected",
    [
        ["az-zen", (1, 0, 0), (0, np.pi / 2, 1)],
        ["ra-dec", (1, 0, 0), (0, 0, 1)],
        ["az-zen", (-1, 0, 0), (np.pi, np.pi / 2, 1)],
        ["ra-dec", (-1, 0, 0), (np.pi, 0, 1)],
        ["az-zen", (0, 1, 0), (np.pi / 2, np.pi / 2, 1)],
        ["ra-dec", (0, 1, 0), (np.pi / 2, 0, 1)],
        ["az-zen", (0, -1, 0), (3 * np.pi / 2, np.pi / 2, 1)],
        ["ra-dec", (0, -1, 0), (3 * np.pi / 2, 0, 1)],
        ["az-zen", (0, 0, 1), (0, 0, 1)],
        ["ra-dec", (0, 0, 1), (0, np.pi / 2, 1)],
        ["az-zen", (0, 0, -1), (0, np.pi, 1)],
        ["ra-dec", (0, 0, -1), (0, -np.pi / 2, 1)],
        ["az-zen", (0, 0, 0), (0, 0, 0)],
        ["ra-dec", (0, 0, 0), (0, 0, 0)],
        ["az-zen", (1, 1, np.sqrt(2)), (np.pi / 4, np.pi / 4, 2)],
        ["ra-dec", (1, 1, np.sqrt(2)), (np.pi / 4, np.pi / 4, 2)],
        ["az-zen", (-1, -1, -np.sqrt(2)), (5 * np.pi / 4, 3 * np.pi / 4, 2)],
        ["ra-dec", (-1, -1, -np.sqrt(2)), (5 * np.pi / 4, -np.pi / 4, 2)],
    ],
)
def test_specific_points_x_prime_to_x_0_2pi(convention, input, expected):
    """Test specific points on a sphere.

    Order is (x, y, z) to (ra/az, dec/zen, r) using [0, 2pi] for ra/az.

    Test:
    - (+/-1, 0, 0) and all permutations
    - (0, 0, 0)
    - (+/-1, +-1, sqrt(2)) a point exactly 2 away from the origin
    """
    parameters = ["a", "b"]
    if convention == "ra-dec":
        prior_bounds = {"a": [0, 2 * np.pi], "b": [-np.pi / 2, np.pi / 2]}
    else:
        prior_bounds = {"a": [0, 2 * np.pi], "b": [0, np.pi]}
    reparam = AnglePair(
        parameters=parameters,
        prior_bounds=prior_bounds,
        convention=convention,
    )

    x_prime = parameters_to_live_point(input, reparam.prime_parameters)
    x = parameters_to_live_point([0, 0, 0], reparam.parameters)
    log_j = 0

    out, _, _ = reparam.inverse_reparameterise(x, x_prime, log_j)

    np.testing.assert_equal(out[reparam.parameters[0]], expected[0])
    np.testing.assert_equal(out[reparam.parameters[1]], expected[1])
    np.testing.assert_equal(out[reparam.parameters[2]], expected[2])


@pytest.mark.parametrize(
    "convention, input, expected",
    [
        ["az-zen", (1, 0, 0), (0, np.pi / 2, 1)],
        ["ra-dec", (1, 0, 0), (0, 0, 1)],
        ["az-zen", (-1, 0, 0), (np.pi, np.pi / 2, 1)],
        ["ra-dec", (-1, 0, 0), (np.pi, 0, 1)],
        ["az-zen", (0, 1, 0), (np.pi / 2, np.pi / 2, 1)],
        ["ra-dec", (0, 1, 0), (np.pi / 2, 0, 1)],
        ["az-zen", (0, -1, 0), (-np.pi / 2, np.pi / 2, 1)],
        ["ra-dec", (0, -1, 0), (-np.pi / 2, 0, 1)],
        ["az-zen", (0, 0, 1), (0, 0, 1)],
        ["ra-dec", (0, 0, 1), (0, np.pi / 2, 1)],
        ["az-zen", (0, 0, -1), (0, np.pi, 1)],
        ["ra-dec", (0, 0, -1), (0, -np.pi / 2, 1)],
        ["az-zen", (0, 0, 0), (0, 0, 0)],
        ["ra-dec", (0, 0, 0), (0, 0, 0)],
        ["az-zen", (1, 1, np.sqrt(2)), (np.pi / 4, np.pi / 4, 2)],
        ["ra-dec", (1, 1, np.sqrt(2)), (np.pi / 4, np.pi / 4, 2)],
        ["az-zen", (-1, -1, -np.sqrt(2)), (-3 * np.pi / 4, 3 * np.pi / 4, 2)],
        ["ra-dec", (-1, -1, -np.sqrt(2)), (-3 * np.pi / 4, -np.pi / 4, 2)],
    ],
)
def test_specific_points_x_prime_to_x_pi_pi(convention, input, expected):
    """Test specific points on a sphere.

    Order is (x, y, z) to (ra/az, dec/zen, r) using [-pi, pi] for ra/az.

    Test:
    - (+/-1, 0, 0) and all permutations
    - (0, 0, 0)
    - (+/-1, +-1, sqrt(2)) a point exactly 2 away from the origin
    """
    parameters = ["a", "b"]
    if convention == "ra-dec":
        prior_bounds = {"a": [-np.pi, np.pi], "b": [-np.pi / 2, np.pi / 2]}
    else:
        prior_bounds = {"a": [-np.pi, np.pi], "b": [0, np.pi]}
    reparam = AnglePair(
        parameters=parameters,
        prior_bounds=prior_bounds,
        convention=convention,
    )

    x_prime = parameters_to_live_point(input, reparam.prime_parameters)
    x = parameters_to_live_point([0, 0, 0], reparam.parameters)
    log_j = 0

    out, _, _ = reparam.inverse_reparameterise(x, x_prime, log_j)

    np.testing.assert_equal(out[reparam.parameters[0]], expected[0])
    np.testing.assert_equal(out[reparam.parameters[1]], expected[1])
    np.testing.assert_equal(out[reparam.parameters[2]], expected[2])


@pytest.mark.parametrize(
    "prior_bounds",
    [
        {"ra": [0, np.pi], "dec": [-np.pi / 2, np.pi / 2]},
        {"ra": [0, 2 * np.pi], "dec": [-np.pi, np.pi]},
    ],
)
def test_invalid_prior_ranges(prior_bounds):
    """Assert an error is raised the prior ranges are invalid"""
    with pytest.raises(ValueError) as excinfo:
        AnglePair(parameters=["ra", "dec"], prior_bounds=prior_bounds)
    assert "Invalid prior ranges" in str(excinfo.value)


def test_invalid_prior_bounds_az():
    """Assert an error is raised the prior bounds are invalid"""
    prior_bounds = {"az": [np.pi, 3 * np.pi], "zen": [0, np.pi]}
    with pytest.raises(ValueError) as excinfo:
        AnglePair(parameters=["az", "zen"], prior_bounds=prior_bounds)
    assert "Prior bounds for az must be" in str(excinfo.value)


def test_invalid_prior_bounds_inc():
    """Assert an error is raised the prior bounds are invalid for the \
        inclination angle.
    """
    prior_bounds = {"ra": [0.0, 2 * np.pi], "dec": [0, np.pi]}
    with pytest.raises(ValueError) as excinfo:
        AnglePair(
            parameters=["ra", "dec"],
            prior_bounds=prior_bounds,
            convention="ra-dec",
        )
    assert "Prior bounds for dec must be" in str(excinfo.value)


def test_unknown_prior():
    """Assert an unknown prior raises an error"""
    with pytest.raises(ValueError) as excinfo:
        AnglePair(
            parameters=["az", "zen"],
            prior_bounds={"az": [0.0, 2 * np.pi], "zen": [0.0, np.pi]},
            convention="az-zen",
            prior="uniform",
        )
    assert "Unknown prior: `uniform`. Choose from: ['isotropic', None]" in str(
        excinfo.value
    )


def test_unknown_convention():
    """Assert an unknown convention raises an error"""
    with pytest.raises(ValueError) as excinfo:
        AnglePair(
            parameters=["az", "zen"],
            prior_bounds={"az": [0.0, 2 * np.pi], "zen": [0.0, np.pi]},
            convention="sky",
        )
    assert (
        "Unknown convention: `sky`. Choose from: ['az-zen', 'ra-dec']"
        in str(excinfo.value)
    )


def test_no_convention():
    """Assert an error is raised if the convention cannot be determined."""
    with pytest.raises(RuntimeError) as excinfo:
        AnglePair(
            parameters=["az", "zen"],
            prior_bounds={"az": [0.0, 2 * np.pi], "zen": [-np.pi, 0.0]},
            convention=None,
        )
    assert "Could not determine convention" in str(excinfo.value)


def test_log_prior():
    """Assert log_prior calls the log pdf of a chi distribution."""
    reparam = create_autospec(AnglePair)
    reparam.parameters = ["a", "b", "r"]
    reparam.has_prior = True
    x = parameters_to_live_point([0, 0, 2.0], reparam.parameters)
    reparam.chi = MagicMock()
    reparam.chi.logpdf = MagicMock(return_value=1.0)
    out = AnglePair.log_prior(reparam, x)
    reparam.chi.assert_not_called()
    reparam.chi.logpdf.assert_called_once_with(x["r"])
    assert out == 1.0


def test_log_prior_radial():
    """Assert log_prior raises an error if chi is False"""
    reparam = create_autospec(AnglePair)
    reparam.chi = False
    reparam.has_prior = True
    with pytest.raises(RuntimeError) as excinfo:
        AnglePair.log_prior(reparam, 1)
    assert "log_prior is not defined" in str(excinfo.value)


def test_log_prior_has_prior():
    """Assert log_prior raises an error if has_prior is False.

    This shouldn't be possible unless the user defines `chi`.
    """
    reparam = create_autospec(AnglePair)
    reparam.chi = True
    reparam.has_prior = False
    with pytest.raises(RuntimeError) as excinfo:
        AnglePair.log_prior(reparam, 1)
    assert "log_prior is not defined" in str(excinfo.value)


def test_x_prime_log_prior():
    """Test the x prime prior"""
    reparam = create_autospec(AnglePair)
    reparam.prime_parameters = ["x", "y", "z"]
    reparam.has_prime_prior = True
    x = parameters_to_live_point([1, 1, 1], reparam.prime_parameters)
    # Should be -1.5 * log(2pi) - 1.5
    expected = -1.5 * (1 + np.log(2 * np.pi))
    out = AnglePair.x_prime_log_prior(reparam, x)
    np.testing.assert_equal(out, expected)


def test_prime_prior_error():
    """Assert an error is raised if calling the prime prior and \
        has prime prior is false.
    """
    reparam = create_autospec(AnglePair)
    reparam.has_prime_prior = False
    with pytest.raises(RuntimeError) as excinfo:
        AnglePair.x_prime_log_prior(reparam, 1)
    assert "x prime prior is not defined" in str(excinfo.value)


def test_reparameterise_negative_radius(reparam):
    """Assert an error is radius if the radius is negative."""
    x = numpy_array_to_live_points(
        np.array([[1.0, 1.0, -1.0]]), ["theta", "phi", "radius"]
    )
    x_prime = x.copy()
    log_j = 0.0
    reparam.radial = "radius"
    reparam.chi = None
    with pytest.raises(RuntimeError) as excinfo:
        AnglePair.reparameterise(reparam, x, x_prime, log_j)
    assert "Radius cannot be negative" in str(excinfo.value)
