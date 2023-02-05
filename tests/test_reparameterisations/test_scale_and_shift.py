# -*- coding: utf-8 -*-
"""
Test the ScaleAndShift class.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, call, create_autospec, patch

from nessai.livepoint import numpy_array_to_live_points
from nessai.reparameterisations import ScaleAndShift
from nessai.utils.testing import assert_structured_arrays_equal


@pytest.fixture()
def reparam():
    return create_autospec(ScaleAndShift)


@pytest.fixture()
def parameters():
    return ["x", "y"]


@pytest.fixture()
def prior_bounds(parameters):
    return {p: [-1, 1] for p in parameters}


@pytest.mark.parametrize("scale", [2, 2.0, [1, 2], {"x": 1, "y": 2}])
def test_init_scale(scale, parameters, prior_bounds):
    """Test the init method with different input types"""
    reparam = ScaleAndShift(
        parameters=parameters, scale=scale, prior_bounds=prior_bounds
    )

    assert not set(reparam.scale.keys()) - set(parameters)
    assert isinstance(reparam.scale["x"], float)
    assert reparam.estimate_scale is False
    assert reparam.estimate_shift is False
    assert reparam._update is False
    assert reparam.shift is None


def test_init_scale_and_shift(reparam, parameters, prior_bounds):
    """Test the init with scale and shift"""
    scale = 1.0
    shift = 2.0
    scale_out = {"x": 1.0, "y": 1.0}
    shift_out = {"x": 2.0, "y": 2.0}
    reparam._check_value = MagicMock(side_effect=[scale_out, shift_out])
    ScaleAndShift.__init__(
        reparam,
        parameters=parameters,
        prior_bounds=prior_bounds,
        scale=scale,
        shift=shift,
    )

    reparam._check_value.assert_has_calls(
        [call(scale, "scale"), call(shift, "shift")]
    )
    assert reparam.scale == scale_out
    assert reparam.shift == shift_out
    assert reparam._update is False


def test_init_estimate(reparam, parameters, prior_bounds):
    """Assert estimate shift and scale are enabled"""
    ScaleAndShift.__init__(
        reparam,
        parameters=parameters,
        prior_bounds=prior_bounds,
        estimate_scale=True,
        estimate_shift=True,
    )
    assert reparam.estimate_scale is True
    assert reparam.estimate_shift is True
    assert list(reparam.scale) == parameters
    assert list(reparam.shift) == parameters
    assert all([v is None for v in reparam.scale.values()])
    assert all([v is None for v in reparam.shift.values()])
    assert reparam._update is True


@pytest.mark.parametrize("n", [1, 2])
def test_reparameterise_scale(reparam, n):
    """Test the reparameterise method"""
    reparam.parameters = ["x", "y"]
    reparam.prime_parameters = ["x_prime", "y_prime"]
    reparam.scale = {"x": -2.0, "y": 4.0}
    reparam.shift = None
    x = numpy_array_to_live_points(np.ones((n, 2)), reparam.parameters)
    x_prime = numpy_array_to_live_points(
        np.zeros((n, 2)), reparam.prime_parameters
    )
    log_j = np.zeros(n)

    x_out, x_prime_out, log_j_out = ScaleAndShift.reparameterise(
        reparam, x, x_prime, log_j
    )

    assert_structured_arrays_equal(x, x_out)
    np.testing.assert_array_equal(log_j_out, -np.log(8 * np.ones(n)))
    assert (x_prime_out["x_prime"] == -0.5).all()
    assert (x_prime_out["y_prime"] == 0.25).all()


@pytest.mark.parametrize("n", [1, 2])
def test_reparameterise_scale_and_shift(reparam, n):
    """Test the reparameterise method"""
    reparam.parameters = ["x", "y"]
    reparam.prime_parameters = ["x_prime", "y_prime"]
    reparam.scale = {"x": -2.0, "y": 4.0}
    reparam.shift = {"x": 2.0, "y": -2.0}
    x = numpy_array_to_live_points(np.ones((n, 2)), reparam.parameters)
    x_prime = numpy_array_to_live_points(
        np.zeros((n, 2)), reparam.prime_parameters
    )
    log_j = np.zeros(n)

    x_out, x_prime_out, log_j_out = ScaleAndShift.reparameterise(
        reparam, x, x_prime, log_j
    )

    assert_structured_arrays_equal(x, x_out)
    np.testing.assert_array_equal(log_j_out, -np.log(8 * np.ones(n)))
    assert (x_prime_out["x_prime"] == 0.5).all()
    assert (x_prime_out["y_prime"] == 0.75).all()


@pytest.mark.parametrize("scale", [1e60, 1e-60])
def test_reparameterise_scale_overflow(reparam, scale):
    """Test the reparameterise method with very small and large scales.

    Checks precision to 14 decimal places.
    """
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.scale = {"x": scale}
    reparam.shift = None
    x_array = np.arange(100.0, dtype=float)
    x = numpy_array_to_live_points(
        scale * x_array[:, np.newaxis], reparam.parameters
    )
    x_prime = numpy_array_to_live_points(
        np.ones((x_array.size, 1)), reparam.prime_parameters
    )
    log_j = np.zeros(x.size)

    x_out, x_prime_out, log_j_out = ScaleAndShift.reparameterise(
        reparam, x, x_prime, log_j
    )

    np.testing.assert_array_almost_equal(
        x_array, x_prime_out["x_prime"], decimal=14
    )
    assert (log_j == -np.log(scale)).all()


@pytest.mark.parametrize("n", [1, 2])
def test_inverse_reparameterise_scale(reparam, n):
    """Test the inverse reparameterise method"""
    reparam.parameters = ["x", "y"]
    reparam.prime_parameters = ["x_prime", "y_prime"]
    reparam.scale = {"x": -2.0, "y": 4.0}
    reparam.shift = None
    x = numpy_array_to_live_points(np.zeros((n, 2)), reparam.parameters)
    x_prime = numpy_array_to_live_points(
        np.ones((n, 2)), reparam.prime_parameters
    )
    x_prime["x_prime"] *= -1
    log_j = np.zeros(n)

    x_out, x_prime_out, log_j_out = ScaleAndShift.inverse_reparameterise(
        reparam, x, x_prime, log_j
    )

    assert_structured_arrays_equal(x_prime, x_prime_out)
    np.testing.assert_array_equal(log_j_out, np.log(8 * np.ones(n)))
    assert (x_out["x"] == 2.0).all()
    assert (x_out["y"] == 4.0).all()


@pytest.mark.parametrize("scale", [1e60, 1e-60])
def test_inverse_reparameterise_scale_overflow(reparam, scale):
    """
    Test the inverse_reparameterise method with very small and large scales.
    """
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.scale = {"x": scale}
    reparam.shift = None
    x_array = np.arange(100.0, dtype=float)
    x = numpy_array_to_live_points(
        np.ones((x_array.size, 1)), reparam.parameters
    )
    x_prime = numpy_array_to_live_points(
        x_array[:, np.newaxis], reparam.prime_parameters
    )
    log_j = np.zeros(x.size)

    x_out, x_prime_out, log_j_out = ScaleAndShift.inverse_reparameterise(
        reparam, x, x_prime, log_j
    )

    np.testing.assert_array_equal(x_array * scale, x_out["x"])
    assert (log_j == np.log(scale)).all()


@pytest.mark.parametrize("n", [1, 2])
def test_inverse_reparameterise_scale_and_shift(reparam, n):
    """Test the inverse reparameterise method"""
    reparam.parameters = ["x", "y"]
    reparam.prime_parameters = ["x_prime", "y_prime"]
    reparam.scale = {"x": -2.0, "y": 4.0}
    reparam.shift = {"x": 1.0, "y": -2.0}
    x = numpy_array_to_live_points(np.zeros((n, 2)), reparam.parameters)
    x_prime = numpy_array_to_live_points(
        np.ones((n, 2)), reparam.prime_parameters
    )
    x_prime["x_prime"] *= -1
    log_j = np.zeros(n)

    x_out, x_prime_out, log_j_out = ScaleAndShift.inverse_reparameterise(
        reparam, x, x_prime, log_j
    )

    assert_structured_arrays_equal(x_prime, x_prime_out)
    np.testing.assert_array_equal(log_j_out, np.log(8 * np.ones(n)))
    assert (x_out["x"] == 3.0).all()
    assert (x_out["y"] == 2.0).all()


@pytest.mark.parametrize("est_scale", [False, True])
@pytest.mark.parametrize("est_shift", [False, True])
def test_update_both(reparam, parameters, est_scale, est_shift):
    """Assert update updates the correct values"""
    x = np.random.randn(4, 2).astype(dtype=[(p, "f8") for p in parameters])
    reparam.parameters = parameters
    reparam.scale = {p: None for p in parameters}
    reparam.shift = {p: None for p in parameters}
    reparam._update = True
    reparam.estimate_scale = est_scale
    reparam.estimate_shift = est_shift
    with patch("numpy.std", side_effect=[1, 2]), patch(
        "numpy.mean", side_effect=[3, 4]
    ):
        ScaleAndShift.update(reparam, x)

    if est_scale:
        assert reparam.scale == dict(zip(parameters, [1, 2]))
    else:
        assert all([v is None for v in reparam.scale.values()])

    if est_shift:
        assert reparam.shift == dict(zip(parameters, [3, 4]))
    else:
        assert all([v is None for v in reparam.shift.values()])


def test_init_no_scale():
    """Make sure an error is raised if the scale is not given"""
    with pytest.raises(
        RuntimeError, match="Must specify a scale or enable estimate_scale"
    ):
        ScaleAndShift(scale=None, estimate_scale=False)


@pytest.mark.parametrize("scale", [[1], [1, 2, 3]])
def test_init_incorrect_scale_list(scale):
    """Make sure an error is raised if the scale is the incorrect length"""
    parameters = ["x", "y"]
    prior_bounds = {"x": [-1, 1], "y": [-1, 1]}

    with pytest.raises(RuntimeError) as excinfo:
        ScaleAndShift(
            parameters=parameters, scale=scale, prior_bounds=prior_bounds
        )

    assert "different length" in str(excinfo.value)


@pytest.mark.parametrize("scale", [{"x": 1}, {"x": 1, "y": 1, "z": 1}])
def test_init_incorrect_scale_dict(scale):
    """Make sure an error is raised if the scale keys to not match the \
            parameters.
    """
    parameters = ["x", "y"]
    prior_bounds = {"x": [-1, 1], "y": [-1, 1]}

    with pytest.raises(RuntimeError) as excinfo:
        ScaleAndShift(
            parameters=parameters, scale=scale, prior_bounds=prior_bounds
        )

    assert "Mismatched parameters" in str(excinfo.value)


def test_init_incorrect_scale_type():
    """Make sure an error is raised if the scale is the incorrect type"""
    parameters = ["x", "y"]
    prior_bounds = {"x": [-1, 1], "y": [-1, 1]}

    with pytest.raises(TypeError, match=r"scale input must be .*"):
        ScaleAndShift(
            parameters=parameters, scale="1", prior_bounds=prior_bounds
        )


def test_init_incorrect_shift_type():
    """Make sure an error is raised if the scale is the incorrect type"""
    parameters = ["x", "y"]
    prior_bounds = {"x": [-1, 1], "y": [-1, 1]}

    with pytest.raises(TypeError, match=r"shift input must be .*"):
        ScaleAndShift(
            parameters=parameters,
            scale=1,
            shift="1",
            prior_bounds=prior_bounds,
        )


@pytest.mark.integration_test
@pytest.mark.parametrize("scale", [-2.0, 2.0])
@pytest.mark.parametrize("shift", [-2.0, 2.0, None])
@pytest.mark.parametrize("estimate_scale", [True, False])
@pytest.mark.parametrize("estimate_shift", [True, False])
def test_invertible(
    is_invertible, model, scale, shift, estimate_scale, estimate_shift
):
    """Assert scale and shift is invertible for different settings."""

    x = model.new_point(100)

    reparam = ScaleAndShift(
        parameters=model.names,
        prior_bounds=model.bounds,
        scale=scale,
        shift=shift,
        estimate_scale=estimate_scale,
        estimate_shift=estimate_shift,
    )

    reparam.update(x)
    assert is_invertible(reparam, atol=1e-14)
