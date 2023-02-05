"""
Test the testing utilities
"""
import numpy as np
import pytest

from nessai.utils.testing import assert_structured_arrays_equal


def test_assert_struct_arrays_different_fields():
    """Assert an errors is raised if the arrays have different fields"""
    x = np.array((1, 2), dtype=[("x", "f8"), ("y", "f8")])
    y = np.array((1, 2), dtype=[("x", "f8"), ("y", "f4")])
    with pytest.raises(AssertionError):
        assert_structured_arrays_equal(x, y)


def test_assert_struct_arrays_equal_values():
    """Assert an errors is raised if fields have different values"""
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f4")]
    x = np.array((1, 2, 3), dtype=dtype)
    y = np.array((1, 3, 4), dtype=dtype)
    with pytest.raises(AssertionError):
        assert_structured_arrays_equal(x, y)


@pytest.mark.parametrize(
    "x, y",
    [
        (
            np.array([(1, 2), (3, 4)], dtype=[("x", "f8"), ("y", "f8")]),
            np.array([(2, 3), (4, 5)], dtype=[("x", "f8"), ("y", "f8")]),
        ),
        (
            np.array([(1, 2), (3, 4)], dtype=[("x", "f8"), ("y", "f8")]),
            np.array(
                [
                    (1, 2),
                ],
                dtype=[("x", "f8"), ("y", "f8")],
            ),
        ),
    ],
)
def test_assert_struct_arrays_equal_array(x, y):
    """Assert an errors is raised with different values with len(x) > 1"""
    with pytest.raises(AssertionError):
        assert_structured_arrays_equal(x, y)


def test_assert_struct_arrays_equal_NaNs():
    """Assert NaNs do not raise an error"""
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f4")]
    x = np.array((1, 2, np.nan), dtype=dtype)
    y = np.array((1, 2, np.nan), dtype=dtype)
    assert_structured_arrays_equal(x, y)


def test_assert_struct_array_equal_tol():
    """Assert arrays that are close do not raised an error if atol is set"""
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8")]
    x = np.array((1.0, 2.0, 3.0), dtype=dtype)
    y = np.array((1.0, 2.0, 3.0 + 1e-10), dtype=dtype)
    assert_structured_arrays_equal(x, y, atol=1e-9, rtol=0.0)


def test_assert_struct_array_equal_tol_error():
    """Assert arrays that are above the atol raised an error"""
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8")]
    x = np.array((1.0, 2.0, 3.0), dtype=dtype)
    y = np.array((1.0, 2.0, 3.0 + 1e-10), dtype=dtype)
    with pytest.raises(AssertionError):
        assert_structured_arrays_equal(x, y, atol=1e-11, rtol=0.0)
