# -*- coding: utf-8 -*-
"""
Test the RescaleToBound class.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, call, create_autospec, patch

from nessai.reparameterisations import RescaleToBounds
from nessai.livepoint import (
    get_dtype,
    empty_structured_array,
    numpy_array_to_live_points,
)
from nessai.utils.testing import assert_structured_arrays_equal

# Tolerances for assert_allclose
atol = 1e-15
rtol = 1e-15


@pytest.fixture
def reparam():
    return create_autospec(RescaleToBounds)


@pytest.fixture()
def reparameterisation(model):
    def _get_reparameterisation(kwargs):
        return RescaleToBounds(
            parameters=model.names, prior_bounds=model.bounds, **kwargs
        )

    return _get_reparameterisation


@pytest.fixture(scope="function")
def is_invertible(model, n=100):
    def test_invertibility(reparam, model=model, decimal=16):
        x = model.new_point(N=n)
        x_prime = empty_structured_array(n, names=reparam.prime_parameters)
        log_j = np.zeros(n)
        assert x.size == x_prime.size

        x_re, x_prime_re, log_j_re = reparam.reparameterise(x, x_prime, log_j)

        x_in = empty_structured_array(x_re.size, reparam.parameters)
        log_j = np.zeros(x_re.size)

        x_inv, x_prime_inv, log_j_inv = reparam.inverse_reparameterise(
            x_in, x_prime_re, log_j
        )

        m = x_re.size // n
        for i in range(m):
            start, end = (i * n), (i + 1) * n
            for name in x.dtype.names:
                np.testing.assert_array_almost_equal(
                    x[name],
                    x_re[name][start:end],
                    decimal=decimal,
                )
                np.testing.assert_array_almost_equal(
                    x[name],
                    x_inv[name][start:end],
                    decimal=decimal,
                )
            for name in x_prime.dtype.names:
                np.testing.assert_array_almost_equal(
                    x_prime_re[name],
                    x_prime_inv[name],
                    decimal=decimal,
                )
            np.testing.assert_array_almost_equal(
                log_j_re, -log_j_inv, decimal=decimal
            )

        return True

    return test_invertibility


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (None, {"x": [-1, 1], "y": [-1, 1]}),
        ([0, 1], {"x": [0, 1], "y": [0, 1]}),
        ({"x": [0, 1], "y": [-1, 1]}, {"x": [0, 1], "y": [-1, 1]}),
    ],
)
def test_rescale_bounds_config(reparam, input, expected_value):
    """Assert the rescale bounds are set correctly."""
    RescaleToBounds.__init__(
        reparam,
        parameters=["x", "y"],
        prior_bounds={"x": [-1, 1], "y": [0, 1]},
        rescale_bounds=input,
    )
    assert reparam.rescale_bounds == expected_value


def test_rescale_bounds_dict_missing_params(reparam):
    """Assert an error is raised if the rescale_bounds dict is missing a
    parameter.
    """
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.__init__(
            reparam,
            parameters=["x", "y"],
            prior_bounds={"x": [-1, 1], "y": [0, 1]},
            rescale_bounds={"x": [0, 1]},
        )
    assert "Missing rescale bounds for parameters" in str(excinfo.value)


def test_rescale_bounds_incorrect_type(reparam):
    """Assert an error is raised if the rescale_bounds is an invalid type."""
    with pytest.raises(TypeError) as excinfo:
        RescaleToBounds.__init__(
            reparam,
            parameters=["x", "y"],
            prior_bounds={"x": [-1, 1], "y": [0, 1]},
            rescale_bounds=1,
        )
    assert "must be an instance of list or dict" in str(excinfo.value)


@pytest.mark.parametrize(
    "input, expected_value",
    [
        (True, {"x": "split", "y": "split"}),
        (False, False),
        (["x"], {"x": "split"}),
        ({"x": "split"}, {"x": "split"}),
        (None, False),
    ],
)
def test_boundary_inversion_config(reparam, input, expected_value):
    """Assert the boundary inversion dict is set correctly"""
    RescaleToBounds.__init__(
        reparam,
        parameters=["x", "y"],
        prior_bounds={"x": [0, 1], "y": [0, 1]},
        boundary_inversion=input,
    )
    assert reparam.boundary_inversion == expected_value


def test_boundary_inversion_invalid_type(reparam):
    """Assert an error is raised in the type is invalid"""
    with pytest.raises(TypeError) as excinfo:
        RescaleToBounds.__init__(
            reparam,
            parameters="x",
            prior_bounds=[0, 1],
            boundary_inversion="Yes",
        )
    assert "boundary_inversion must be a list, dict or bool" in str(
        excinfo.value
    )


def test_detect_edges_without_inversion(reparam):
    """Assert detect edges cannot be used with boundary inversion"""
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.__init__(
            reparam,
            parameters=["x", "y"],
            prior_bounds={"x": [-1, 1], "y": [0, 1]},
            detect_edges=True,
        )
    assert "Must enable boundary inversion to use detect edges" in str(
        excinfo.value
    )


def test_set_bounds(reparam):
    """Test the set bounds method."""
    reparam.parameters = ["x"]
    reparam.rescale_bounds = {"x": np.array([-1, 1])}
    reparam.pre_rescaling = lambda x: (x / 2, np.zeros_like(x))
    reparam.offsets = {"x": 1}
    RescaleToBounds.set_bounds(reparam, {"x": np.array([-10, 10])})
    np.testing.assert_array_equal(reparam.pre_prior_bounds["x"], [-5, 5])
    np.testing.assert_array_equal(reparam.bounds["x"], [-6, 4])


def test_set_offets(reparam):
    """Assert the offset are set correctly"""
    reparam.pre_rescaling = lambda x: (x / 2, 0.0)

    RescaleToBounds.__init__(
        reparam,
        parameters=["x", "y"],
        prior_bounds={"x": [8, 32], "y": [2, 4]},
        offset=True,
    )

    assert reparam.offsets == {"x": 10.0, "y": 1.5}


def test_reset_inversion(reparam):
    """Assert the edges are reset correctly"""
    reparam.parameters = ["x", "y"]
    reparam._edges = {"x": [-10, 10], "y": [-5, 5]}
    RescaleToBounds.reset_inversion(reparam)
    assert reparam._edges == {"x": None, "y": None}


def test_reset_inversion_no_edges(reparam):
    """Assert the edges are reset not reset if _edges is None"""
    reparam.parameters = ["x", "y"]
    reparam._edges = None
    RescaleToBounds.reset_inversion(reparam)
    assert reparam._edges is None


def test_update(reparam):
    """Assert update calls the correct methods"""
    reparam.reset_inversion = MagicMock()
    reparam.update_bounds = MagicMock()
    x = np.array((1, 2), dtype=[("x", "f8"), ("y", "f8")])
    RescaleToBounds.update(reparam, x)
    reparam.reset_inversion.assert_called_once()
    reparam.update_bounds.assert_called_once_with(x)


def test_x_prime_log_prior_error(reparam):
    """Assert an error is raised if the prime prior is not defined."""
    reparam.has_prime_prior = False
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.x_prime_log_prior(reparam, 0.1)
    assert "Prime prior is not configured" in str(excinfo.value)


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

    rescaling = "logit"
    RescaleToBounds.configure_pre_rescaling(reparam, rescaling)
    assert reparam.has_pre_rescaling is True
    assert reparam.pre_rescaling is rescaling_functions["logit"][0]
    assert reparam.pre_rescaling_inv is rescaling_functions["logit"][1]


def test_pre_rescaling_with_invalid_str(reparam):
    """Assert an error is raised if the rescaling is not recognised"""
    rescaling = "not_a_rescaling"
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.configure_pre_rescaling(reparam, rescaling)
    assert "Unknown rescaling function: not_a_rescaling" in str(excinfo.value)


def test_post_rescaling_with_invalid_str(reparam):
    """Assert an error is raised if the rescaling is not recognised"""
    rescaling = "not_a_rescaling"
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.configure_post_rescaling(reparam, rescaling)
    assert "Unknown rescaling function: not_a_rescaling" in str(excinfo.value)


def test_post_rescaling_with_str(reparam):
    """Assert that specifying a str works as intended.

    Also test the config for the logit
    """
    reparam._update_bounds = False
    reparam.parameters = ["x"]
    from nessai.utils.rescaling import rescaling_functions

    rescaling = "logit"
    RescaleToBounds.configure_post_rescaling(reparam, rescaling)
    assert reparam.has_post_rescaling is True
    assert reparam.has_prime_prior is False
    assert reparam.post_rescaling is rescaling_functions["logit"][0]
    assert reparam.post_rescaling_inv is rescaling_functions["logit"][1]
    assert reparam.rescale_bounds == {"x": [0, 1]}


def test_post_rescaling_with_logit_update_bounds(reparam):
    """Assert an error is raised if using logit and update bounds"""
    reparam._update_bounds = True
    rescaling = "logit"
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.configure_post_rescaling(reparam, rescaling)
    assert "Cannot use logit with update bounds" in str(excinfo.value)


def test_pre_rescaling_invalid_input(reparam):
    """Assert an error is raised if the input isn't a str or tuple"""
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.configure_pre_rescaling(reparam, (np.exp,))
    assert "Pre-rescaling must be a str or tuple" in str(excinfo.value)


def test_post_rescaling_invalid_input(reparam):
    """Assert an error is raised if the input isn't a str or tuple"""
    with pytest.raises(RuntimeError) as excinfo:
        RescaleToBounds.configure_post_rescaling(reparam, (np.exp,))
    assert "Post-rescaling must be a str or tuple" in str(excinfo.value)


def test_update_bounds_disabled(reparam, caplog):
    """Assert nothing happens in _update_bounds is False"""
    caplog.set_level("DEBUG")
    reparam._update_bounds = False
    RescaleToBounds.update_bounds(reparam, [0, 1])
    assert "Update bounds not enabled" in str(caplog.text)


def test_update_bounds(reparam):
    """Assert the correct values are returned"""
    reparam.offsets = {"x": 0.0, "y": 1.0}
    reparam.pre_rescaling = MagicMock(
        side_effect=lambda x: (x, np.zeros_like(x))
    )
    reparam.parameters = ["x", "y"]
    x = {"x": [-1, 0, 1], "y": [-2, 0, 2]}
    RescaleToBounds.update_bounds(reparam, x)
    reparam.update_prime_prior_bounds.assert_called_once()
    reparam.pre_rescaling.assert_has_calls(
        [call(-1), call(1), call(-2), call(2)]
    )
    assert reparam.bounds == {"x": [-1, 1], "y": [-3, 1]}


def test_reparameterise(reparam):
    """Test the reparameterise function"""
    reparam.has_pre_rescaling = False
    reparam.has_post_rescaling = False
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.offsets = {"x": 1.0}
    reparam.boundary_inversion = False
    x = numpy_array_to_live_points(
        np.array([(1.0,), (2.0,)]), reparam.parameters
    )
    x_prime_in = np.zeros(
        [
            2,
        ],
        dtype=get_dtype(reparam.prime_parameters),
    )
    x_prime_val = np.array([0.0, 0.5])
    log_j = np.zeros(x.size)

    reparam._rescale_to_bounds = MagicMock(
        return_value=(x_prime_val, np.array([0, 0.5]))
    )

    x_out, x_prime_out, log_j_out = RescaleToBounds.reparameterise(
        reparam, x, x_prime_in, log_j
    )

    np.testing.assert_array_equal(
        np.array([0.0, 1.0]),
        reparam._rescale_to_bounds.call_args_list[0][0][0],
    )

    assert reparam._rescale_to_bounds.call_args_list[0][0][1] == "x"

    assert_structured_arrays_equal(x, x_out)
    np.testing.assert_array_equal(x_prime_out["x_prime"], x_prime_val)
    np.testing.assert_array_equal(log_j_out, np.array([0.0, 0.5]))


def test_inverse_reparameterise(reparam):
    """Test the inverse_reparameterise function"""
    reparam.has_pre_rescaling = False
    reparam.has_post_rescaling = False
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.offsets = {"x": 1.0}
    reparam.boundary_inversion = False
    x_prime = numpy_array_to_live_points(
        np.array([(1.0,), (2.0,)]), reparam.prime_parameters
    )
    x_in = np.zeros(
        [
            2,
        ],
        dtype=get_dtype(reparam.parameters),
    )
    x_val = np.array([0.0, 0.5])
    log_j = np.zeros(x_prime.size)

    reparam._inverse_rescale_to_bounds = MagicMock(
        return_value=(x_val, np.array([0, 0.5]))
    )

    x_out, x_prime_out, log_j_out = RescaleToBounds.inverse_reparameterise(
        reparam, x_in, x_prime, log_j
    )

    # x[p] is updated in place, can't test inputs
    reparam._inverse_rescale_to_bounds.assert_called_once()
    assert reparam._inverse_rescale_to_bounds.call_args_list[0][0][1] == "x"

    assert_structured_arrays_equal(x_prime_out, x_prime)
    np.testing.assert_array_equal(x_out["x"], x_val + 1.0)
    np.testing.assert_array_equal(log_j_out, np.array([0.0, 0.5]))


def test_reparameterise_boundary_inversion(reparam):
    """Test the reparameterise function with boundary inversion"""
    reparam.has_pre_rescaling = False
    reparam.has_post_rescaling = False
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.offsets = {"x": 1.0}
    reparam.boundary_inversion = {"x": "split"}
    x = numpy_array_to_live_points(
        np.array([(1.0,), (2.0,)]), reparam.parameters
    )
    inversion_out = numpy_array_to_live_points(
        np.array([(-1.0,), (-2.0,), (1.0,), (2.0,)]), reparam.prime_parameters
    )
    x_prime_in = np.zeros(
        [
            2,
        ],
        dtype=get_dtype(reparam.prime_parameters),
    )
    log_j = np.zeros(x.size)

    x_ex = np.concatenate([x, x])
    x_prime_ex = inversion_out
    log_j_ex = np.array([0, 0.5, 0, 0.5])

    reparam._apply_inversion = MagicMock(
        return_value=(x_ex, x_prime_ex, log_j_ex)
    )

    x_out, x_prime_out, log_j_out = RescaleToBounds.reparameterise(
        reparam,
        x,
        x_prime_in,
        log_j,
        compute_radius=True,
        test="test",
    )

    assert_structured_arrays_equal(
        reparam._apply_inversion.call_args_list[0][0][0], x
    )
    assert_structured_arrays_equal(
        reparam._apply_inversion.call_args_list[0][0][1],
        x_prime_in,
    )
    np.testing.assert_array_equal(
        reparam._apply_inversion.call_args_list[0][0][2], log_j
    )
    assert reparam._apply_inversion.call_args_list[0][0][3] == "x"
    assert reparam._apply_inversion.call_args_list[0][0][4] == "x_prime"
    assert reparam._apply_inversion.call_args_list[0][0][5] is True
    assert reparam._apply_inversion.call_args_list[0][1] == {"test": "test"}

    assert_structured_arrays_equal(x_out, x_ex)
    assert_structured_arrays_equal(x_prime_out, x_prime_ex)
    np.testing.assert_array_equal(log_j_out, log_j_ex)


def test_inverse_reparameterise_boundary_inversion(reparam):
    """Test the inverse_reparameterise function with boundary inversion"""
    reparam.has_pre_rescaling = False
    reparam.has_post_rescaling = False
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x"]
    reparam.offsets = {"x": 1.0}
    reparam.boundary_inversion = {"x": "split"}
    x_prime = numpy_array_to_live_points(
        np.array([(-1.0,), (2.0,)]), reparam.prime_parameters
    )
    inversion_out = numpy_array_to_live_points(
        np.array([(1.0,), (2.0,)]), reparam.parameters
    )
    x_in = np.zeros(
        [
            2,
        ],
        dtype=get_dtype(reparam.parameters),
    )
    log_j = np.zeros(x_prime.size)

    x_ex = inversion_out
    x_prime_ex = x_prime
    log_j_ex = np.array([0, 0.5])

    reparam._reverse_inversion = MagicMock(
        return_value=(x_ex, x_prime_ex, log_j_ex)
    )

    x_out, x_prime_out, log_j_out = RescaleToBounds.inverse_reparameterise(
        reparam,
        x_in,
        x_prime,
        log_j,
    )

    assert_structured_arrays_equal(
        reparam._reverse_inversion.call_args_list[0][0][0], x_in
    )
    assert_structured_arrays_equal(
        reparam._reverse_inversion.call_args_list[0][0][1], x_prime
    )
    np.testing.assert_array_equal(
        reparam._reverse_inversion.call_args_list[0][0][2], log_j
    )

    assert_structured_arrays_equal(x_out, x_ex)
    assert_structured_arrays_equal(x_prime_out, x_prime_ex)
    np.testing.assert_array_equal(log_j_out, log_j_ex)


def test_reparameterise_pre_post_rescaling(reparam):
    """Test the reparameterise function with pre and post rescaling"""
    reparam.has_pre_rescaling = True
    reparam.has_post_rescaling = True
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.offsets = {"x": 0.0}
    reparam.boundary_inversion = {}
    x = numpy_array_to_live_points(
        np.array([(1.0,), (2.0,)]), reparam.parameters
    )
    x_prime_in = np.zeros(
        [
            2,
        ],
        dtype=get_dtype(reparam.prime_parameters),
    )
    x_prime_val = np.array([4.0, 8.0])
    log_j = np.zeros(x.size)

    reparam._rescale_to_bounds = MagicMock(
        return_value=(x_prime_val, np.array([0, 0.5]))
    )
    reparam.pre_rescaling = MagicMock(
        return_value=(np.array([0.5, 1.0]), np.array([0.5, 0.5]))
    )
    reparam.post_rescaling = MagicMock(
        return_value=(np.array([1.0, 1.5]), np.array([2.0, 3.0]))
    )

    x_out, x_prime_out, log_j_out = RescaleToBounds.reparameterise(
        reparam, x, x_prime_in, log_j
    )

    np.testing.assert_array_equal(
        np.array([0.5, 1.0]),
        reparam._rescale_to_bounds.call_args_list[0][0][0],
    )

    assert reparam._rescale_to_bounds.call_args_list[0][0][1] == "x"

    np.testing.assert_array_equal(
        reparam.pre_rescaling.call_args_list[0][0][0],
        np.array([1.0, 2.0]),
    )
    # x_prime gets replaces so change check inputs to the function
    reparam.post_rescaling.assert_called_once()

    assert_structured_arrays_equal(x, x_out)
    np.testing.assert_array_equal(x_prime_out["x_prime"], np.array([1.0, 1.5]))
    np.testing.assert_array_equal(log_j_out, np.array([2.5, 4.0]))


def test_inverse_reparameterise_pre_post_rescaling(reparam):
    """Test the inverse_reparameterise function with pre and post rescaling"""
    reparam.has_pre_rescaling = True
    reparam.has_post_rescaling = True
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.offsets = {"x": 0.0}
    reparam.boundary_inversion = {}
    x_prime = numpy_array_to_live_points(
        np.array([(1.0,), (2.0,)]), reparam.prime_parameters
    )
    x_in = np.zeros(
        [
            2,
        ],
        dtype=get_dtype(reparam.parameters),
    )
    x_val = np.array([4.0, 8.0])
    log_j = np.zeros(x_prime.size)

    reparam._inverse_rescale_to_bounds = MagicMock(
        return_value=(x_val, np.array([0, 0.5]))
    )
    reparam.pre_rescaling_inv = MagicMock(
        return_value=(np.array([0.5, 1.0]), np.array([0.5, 0.5]))
    )
    reparam.post_rescaling_inv = MagicMock(
        return_value=(np.array([1.0, 1.5]), np.array([2.0, 3.0]))
    )

    x_out, x_prime_out, log_j_out = RescaleToBounds.inverse_reparameterise(
        reparam, x_in, x_prime, log_j
    )

    np.testing.assert_array_equal(
        np.array([0.5, 1.0]),
        reparam._inverse_rescale_to_bounds.call_args_list[0][0][0],
    )

    assert reparam._inverse_rescale_to_bounds.call_args_list[0][0][1] == "x"

    np.testing.assert_array_equal(
        reparam.post_rescaling_inv.call_args_list[0][0][0],
        x_prime["x_prime"],
    )
    reparam.pre_rescaling_inv.assert_called_once()
    # x_prime gets replaces so change check inputs to the function
    reparam.post_rescaling_inv.assert_called_once()

    assert_structured_arrays_equal(x_prime, x_prime_out)
    np.testing.assert_array_equal(x_out["x"], np.array([0.5, 1.0]))
    np.testing.assert_array_equal(log_j_out, np.array([2.5, 4.0]))


def test_apply_inversion_detect_edge(reparam):
    """Assert detect edge is called with the correct arguments"""
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.offsets = {"x": 1.0}
    reparam._edges = {"x": None}
    reparam.detect_edges_kwargs = {"allowed_bounds": ["lower"]}
    reparam.bounds = {"x": [0, 5]}
    reparam.rescale_bounds = {"x": [0, 1]}

    x = numpy_array_to_live_points(np.array([1, 2]), ["x"])
    x_prime = numpy_array_to_live_points(np.array([3, 4]), ["x_prime"])
    log_j = np.zeros(2)

    with patch(
        "nessai.reparameterisations.rescale.detect_edge", return_value=False
    ) as mock_fn:

        _ = RescaleToBounds._apply_inversion(
            reparam, x, x_prime, log_j, "x", "x_prime", False, test=True
        )

    reparam.update_prime_prior_bounds.assert_called_once()
    mock_fn.assert_called_once_with(
        x_prime["x_prime"],
        test=True,
        allowed_bounds=["lower"],
    )

    assert reparam._edges == {"x": False}


def test_apply_inversion_not_applied(reparam):
    """Assert the apply inversion works correctly"""
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.offsets = {"x": 1.0}
    reparam._edges = {"x": False}
    reparam.bounds = {"x": [0, 5]}
    reparam.rescale_bounds = {"x": [0, 1]}

    x_val = np.array([[1], [2]])
    x_prime = numpy_array_to_live_points(x_val, ["x_prime"])
    x = numpy_array_to_live_points(np.array([3, 4]), ["x"])
    log_j = np.zeros(2)

    with patch(
        "nessai.reparameterisations.rescale.rescale_minus_one_to_one",
        side_effect=lambda x, *args, **kwargs: (x, np.array([5, 6])),
    ) as f:
        x_out, x_prime_out, log_j_out = RescaleToBounds._apply_inversion(
            reparam,
            x,
            x_prime,
            log_j,
            "x",
            "x_prime",
            False,
        )

    assert f.call_args_list[0][1] == {"xmin": 0, "xmax": 5}
    # Should be output of rescaling minus offset
    np.testing.assert_array_equal(x_prime_out["x_prime"], np.array([0, 1]))
    # x_prime should be the same
    assert x_out is x
    # Jacobian should just include jacobian from rescaling
    np.testing.assert_array_equal(log_j_out, np.array([5, 6]))


def test_apply_inversion_split(reparam):
    """Assert apply inversion with split works as intended.

    Also tests "upper" setting.
    """
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.offsets = {"x": 1.0}
    reparam._edges = {"x": "upper"}
    reparam.bounds = {"x": [0, 5]}
    reparam.rescale_bounds = {"x": [0, 1]}
    reparam.boundary_inversion = {"x": "split"}

    x_val = np.array([[1.2], [1.7]])
    x_prime = numpy_array_to_live_points(x_val, ["x_prime"])
    x = numpy_array_to_live_points(np.array([3, 4]), ["x"])
    log_j = np.zeros(2)

    with patch(
        "numpy.random.choice", return_value=np.array([1])
    ) as rnd, patch(
        "nessai.reparameterisations.rescale.rescale_zero_to_one",
        side_effect=lambda x, *args: (x, np.array([5, 6])),
    ) as f:
        x_out, x_prime_out, log_j_out = RescaleToBounds._apply_inversion(
            reparam,
            x,
            x_prime,
            log_j,
            "x",
            "x_prime",
            False,
        )

    rnd.assert_called_once_with(2, 1, replace=False)
    assert f.call_args_list[0][0][1] == 0.0
    assert f.call_args_list[0][0][2] == 5.0
    # Output should be x_val minus offset
    # Then 1 - that for 'upper'
    # Then *= -1 with the output of rnd
    np.testing.assert_array_almost_equal(
        x_prime_out["x_prime"],
        np.array([0.8, -0.3]),
        decimal=10,
    )
    # x should be the same
    assert x_out is x
    # Jacobian should just include jacobian from rescaling
    np.testing.assert_array_equal(log_j_out, np.array([5, 6]))


@pytest.mark.parametrize(
    "inv_type, compute_radius",
    [("duplicate", False), ("duplicate", True), ("split", True)],
)
def test_apply_inversion_duplicate(reparam, inv_type, compute_radius):
    """Assert apply inversion with duplicate works as intended.

    This test also covers compute_radius=True
    """
    reparam.parameters = ["x", "y"]
    reparam.prime_parameters = ["x_prime", "y"]
    reparam.offsets = {"x": 1.0}
    reparam._edges = {"x": "lower"}
    reparam.bounds = {"x": [0, 5]}
    reparam.rescale_bounds = {"x": [0, 1]}
    reparam.boundary_inversion = {"x": inv_type}

    x_val = np.array([[1.2, 1.0], [1.7, 2.0]])
    x_prime = numpy_array_to_live_points(x_val, ["x_prime", "y"])
    x = numpy_array_to_live_points(np.array([[1, 2], [3, 4]]), ["x", "y"])
    log_j = np.zeros(2)

    with patch("numpy.random.choice") as rnd, patch(
        "nessai.reparameterisations.rescale.rescale_zero_to_one",
        side_effect=lambda x, *args: (x, np.array([5, 6])),
    ) as f:
        x_out, x_prime_out, log_j_out = RescaleToBounds._apply_inversion(
            reparam,
            x,
            x_prime,
            log_j,
            "x",
            "x_prime",
            compute_radius,
        )

    rnd.assert_not_called()
    assert f.call_args_list[0][0][1] == 0.0
    assert f.call_args_list[0][0][2] == 5.0
    # Output should be x_val minus offset
    # Then duplicated
    np.testing.assert_array_almost_equal(
        x_prime_out["x_prime"],
        np.array([0.2, 0.7, -0.2, -0.7]),
        decimal=10,
    )
    np.testing.assert_array_almost_equal(
        x_prime_out["y"],
        np.array([1.0, 2.0, 1.0, 2.0]),
        decimal=10,
    )
    # x should be the same but duplicated
    assert_structured_arrays_equal(x_out, np.concatenate([x, x]))
    # Jacobian should just include jacobian from rescaling but duplicated
    np.testing.assert_array_equal(log_j_out, np.array([5, 6, 5, 6]))


def test_reverse_inversion(reparam):
    """Assert the reverse inversion works correctly"""
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.offsets = {"x": 1.0}
    reparam._edges = {"x": "upper"}
    reparam.bounds = {"x": [0, 5]}

    x_val = np.array([[-0.7], [0.4]])
    x = numpy_array_to_live_points(x_val, ["x"])
    x_prime = numpy_array_to_live_points(np.array([3, 4]), ["x_prime"])
    log_j = np.zeros(2)

    # Return the same value to check that the negative values are handled
    # correctly
    with patch(
        "nessai.reparameterisations.rescale.inverse_rescale_zero_to_one",
        side_effect=lambda x, *args: (x, np.array([5, 6])),
    ) as f:
        x_out, x_prime_out, log_j_out = RescaleToBounds._reverse_inversion(
            reparam,
            x,
            x_prime,
            log_j,
            "x",
            "x_prime",
        )

    assert f.call_args_list[0][0][1] == 0.0
    assert f.call_args_list[0][0][2] == 5.0
    # Should be output of rescaling minus offset
    np.testing.assert_array_equal(x_out["x"], np.array([1.3, 1.6]))
    # x_prime should be the same
    assert x_prime_out is x_prime
    # Jacobian should just include jacobian from rescaling
    np.testing.assert_array_equal(log_j_out, np.array([5, 6]))


def test_reverse_inversion_not_applied(reparam):
    """Assert the reverse inversion works correctly"""
    reparam.parameters = ["x"]
    reparam.prime_parameters = ["x_prime"]
    reparam.offsets = {"x": 1.0}
    reparam._edges = {"x": False}
    reparam.bounds = {"x": [0, 5]}

    x_val = np.array([[1], [2]])
    x = numpy_array_to_live_points(x_val, ["x"])
    x_prime = numpy_array_to_live_points(np.array([3, 4]), ["x_prime"])
    log_j = np.zeros(2)

    with patch(
        "nessai.reparameterisations.rescale.inverse_rescale_minus_one_to_one",
        side_effect=lambda x, *args, **kwargs: (x, np.array([5, 6])),
    ) as f:
        x_out, x_prime_out, log_j_out = RescaleToBounds._reverse_inversion(
            reparam,
            x,
            x_prime,
            log_j,
            "x",
            "x_prime",
        )

    assert f.call_args_list[0][1] == {"xmin": 0, "xmax": 5}
    # Should be output of rescaling minus offset
    np.testing.assert_array_equal(x_out["x"], np.array([2, 3]))
    # x_prime should be the same
    assert x_prime_out is x_prime
    # Jacobian should just include jacobian from rescaling
    np.testing.assert_array_equal(log_j_out, np.array([5, 6]))


@pytest.mark.parametrize(
    "rescale_bounds", [None, [0, 1], {"x": [0, 1], "y": [-1, 1]}]
)
@pytest.mark.integration_test
def test_rescale_bounds(reparameterisation, is_invertible, rescale_bounds):
    """Test the different options for rescale to bounds"""
    reparam = reparameterisation({"rescale_bounds": rescale_bounds})
    if rescale_bounds is None:
        rescale_bounds = {p: [-1, 1] for p in reparam.parameters}
    elif isinstance(rescale_bounds, list):
        rescale_bounds = {p: rescale_bounds for p in reparam.parameters}

    assert reparam.rescale_bounds == rescale_bounds
    assert is_invertible(reparam)


@pytest.mark.parametrize(
    "boundary_inversion",
    [False, True, ["x"], {"x": "split"}, {"x": "duplicate"}],
)
@pytest.mark.integration_test
def test_boundary_inversion(
    reparameterisation, is_invertible, boundary_inversion
):
    """Test the different options for rescale to bounds"""
    reparam = reparameterisation({"boundary_inversion": boundary_inversion})

    assert is_invertible(reparam)


@pytest.mark.integration_test
def test_update_prime_prior_bounds_integration():
    """Assert the prime prior bounds are correctly computed"""
    rescaling = (
        lambda x: (x / 2, np.zeros_like(x)),
        lambda x: (2 * x, np.zeros_like(x)),
    )
    reparam = RescaleToBounds(
        parameters=["x"],
        prior_bounds=[1000, 1001],
        prior="uniform",
        pre_rescaling=rescaling,
        offset=True,
    )
    np.testing.assert_equal(reparam.offsets["x"], 500.25)
    np.testing.assert_array_equal(reparam.prior_bounds["x"], [1000, 1001])
    np.testing.assert_array_equal(reparam.pre_prior_bounds["x"], [500, 500.5])
    np.testing.assert_array_equal(reparam.bounds["x"], [-0.25, 0.25])
    np.testing.assert_array_equal(
        reparam.prime_prior_bounds["x_prime"], [-1, 1]
    )

    x_prime = numpy_array_to_live_points(
        np.array([[-2], [-1], [0.5], [1], [10]]), ["x_prime"]
    )
    log_prior = reparam.x_prime_log_prior(x_prime)
    expected = np.array([-np.inf, 0, 0, 0, -np.inf])
    np.testing.assert_equal(log_prior, expected)


@pytest.mark.integration_test
def test_pre_rescaling_integration(is_invertible, model):
    """Test the pre-scaling feature"""

    def forward(x):
        return np.log(x), -np.log(x)

    def inv(x):
        return np.exp(x), x.copy()

    reparam = RescaleToBounds(
        parameters="x",
        prior_bounds={"x": [1.0, np.e]},
        pre_rescaling=(forward, inv),
        rescale_bounds=[-1.0, 1.0],
    )

    x = numpy_array_to_live_points(
        np.array([[1.0], [np.e**0.5], [2.0], [np.e]]), ["x"]
    )
    x_prime = empty_structured_array(x.size, ["x_prime"])
    log_j = np.zeros(x.size)

    x_out, x_prime_out, log_j_out = reparam.reparameterise(x, x_prime, log_j)

    assert_structured_arrays_equal(x_out, x)
    np.testing.assert_allclose(
        x_prime_out["x_prime"],
        np.array([-1, 0.0, 2 * np.log(2) - 1, 1]),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        log_j_out,
        -np.log(x["x"]) + np.log(2),
        rtol=rtol,
        atol=atol,
    )

    x_in = empty_structured_array(x_prime_out.size, ["x"])
    log_j = np.zeros(x.size)
    x_out, x_prime_final, log_j_final = reparam.inverse_reparameterise(
        x_in, x_prime_out, log_j
    )

    np.testing.assert_allclose(
        log_j_final,
        np.log(x_out["x"]) - np.log(2),
        rtol=rtol,
        atol=atol,
    )

    np.testing.assert_allclose(x_out["x"], x["x"], rtol=rtol, atol=atol)
    assert_structured_arrays_equal(x_prime_final, x_prime_out)
    np.testing.assert_allclose(log_j_final, -log_j_out, rtol=rtol, atol=atol)

    # Trick to get a 1d model to test only x
    model._names.remove("y")
    model._bounds = {"x": [1.0, np.e]}
    assert is_invertible(reparam, model=model, decimal=12)


@pytest.mark.integration_test
def test_update_integration(model):
    """Assert edges and bounds are updated"""

    x = model.new_point(2)

    new_bounds = {"x": [np.min(x["x"]), np.max(x["x"])]}

    reparam = RescaleToBounds(
        parameters="x",
        update_bounds=True,
        boundary_inversion=True,
        detect_edges=True,
        prior_bounds=model.bounds["x"],
    )

    reparam._edges = {"x": "lower"}
    reparam.bounds = {"x": [-100, 100]}

    reparam.update(x)

    assert reparam._edges == {"x": None}
    assert reparam.bounds == new_bounds


@pytest.mark.integration_test
def test_update_integration_no_update(model):
    """Assert the bounds and edges are not updated if disabled."""
    x = model.new_point(2)

    reparam = RescaleToBounds(
        parameters="x",
        update_bounds=False,
        boundary_inversion=False,
        detect_edges=False,
        prior_bounds=model.bounds["x"],
    )

    reparam.update(x)

    assert reparam._edges is None
    np.testing.assert_array_equal(reparam.bounds["x"], model.bounds["x"])


@pytest.mark.parametrize(
    "kwargs, decimal",
    [
        (dict(post_rescaling="logit", update_bounds=False), 10),
        (dict(update_bounds=False), None),
        (dict(update_bounds=False, boundary_inversion=True), None),
        (dict(boundary_inversion=["x"]), None),
    ],
)
@pytest.mark.integration_test
def test_is_invertible_general_config(is_invertible, model, kwargs, decimal):
    """Test the invertibility of the reparameterisation

    General tests that don't check specific attributes but can check
    combinations.
    """
    if decimal is None:
        decimal = 16
    default_kwargs = dict(
        parameters=model.names,
        prior_bounds=model.bounds,
    )
    default_kwargs.update(kwargs)
    reparam = RescaleToBounds(**default_kwargs)
    assert is_invertible(reparam, decimal=decimal)
