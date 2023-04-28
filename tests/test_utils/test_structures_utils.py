# -*- coding: utf-8 -*-
"""
Tests for utilities related to python structures such as lists.
"""
import numpy as np
import pytest

from nessai.utils.structures import (
    array_split_chunksize,
    get_inverse_indices,
    get_subset_arrays,
    isfinite_struct,
    replace_in_list,
)


def test_replace_in_list():
    """
    Test if the list produced contains the correct entries in the correct
    locations
    """
    x = [1, 2, 3]
    replace_in_list(x, [1, 2], [5, 4])
    assert x == [5, 4, 3]


def test_replace_in_list_item():
    """
    Test if items are correctly converted to lists in replace_in_list function
    """
    x = [1, 2, 3]
    replace_in_list(x, 3, 4)
    assert x == [1, 2, 4]


def test_replace_in_list_single():
    """Test the function with a list of strings"""
    x = ["aa", "bb", "cc"]
    replace_in_list(x, ["aa", "bb"], ["dd", "ee"])
    assert x == ["dd", "ee", "cc"]


def test_replace_in_list_single_str():
    """Test the function with single strings"""
    x = ["aa", "bb", "cc"]
    replace_in_list(x, "aa", "dd")
    assert x == ["dd", "bb", "cc"]


def test_different_lengths():
    """
    Assert an error is raised if the targets and replacements are different
    lengths.
    """
    with pytest.raises(RuntimeError) as excinfo:
        replace_in_list([1, 2], [1, 2], 3)
    assert "Targets and replacements are different lengths!" in str(
        excinfo.value
    )


def test_missing_targets():
    """Assert an error is raised if a target is not in the target list."""
    with pytest.raises(ValueError) as excinfo:
        replace_in_list([1, 2], 4, 3)
    assert "Targets [4] not in list: [1, 2]" in str(excinfo.value)


def test_get_subset_arrays():
    """Assert the correct subsets are returned."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    indices = np.array([1, 2])
    a_out, b_out = get_subset_arrays(indices, a, b)
    np.testing.assert_equal(a_out, a[indices])
    np.testing.assert_equal(b_out, b[indices])


def test_get_subset_arrays_empty():
    """Assert output is empty if no arrays are provided"""
    out = get_subset_arrays(np.array([1, 2]))
    assert out == ()


@pytest.mark.parametrize(
    "names, expected",
    [
        (None, [False, True, False]),
        (["x", "y", "z"], [False, True, False]),
        (["x"], [True, True, True]),
        (["y"], [False, True, True]),
        (["z"], [True, True, False]),
    ],
)
def test_isfinite_struct(names, expected):
    """Assert the correct array is returned"""
    x = np.array(
        [(0, np.inf, 0), (1, 1, 1), (2, 2, np.nan)],
        dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")],
    )
    out = isfinite_struct(x, names=names)
    assert len(out) == 3
    np.testing.assert_equal(out, np.array(expected))


@pytest.mark.integration_test
def test_isfinite_struct_invalid_name():
    """Assert an error is raised if a name is invalid.

    Numpy should raise a ValueError if the name is not a field in the array.
    """
    x = np.array([(1,), (2,)], dtype=[("x", "i4")])
    with pytest.raises(ValueError):
        isfinite_struct(x, ["y"])


def test_array_split_chunksize():
    """Assert the correct array sizes are returned"""
    a = np.array([1, 2, 3, 4, 5])
    out = array_split_chunksize(a, 2)
    assert len(out) == 3
    np.testing.assert_array_equal(out[0], a[:2])
    np.testing.assert_array_equal(out[1], a[2:4])
    np.testing.assert_array_equal(out[2], a[4:])


def test_array_split_chunksize_larger_than_array():
    """Assert the correct array sizes are returned"""
    a = np.array([1, 2, 3, 4, 5])
    out = array_split_chunksize(a, 6)
    assert len(out) == 1
    np.testing.assert_array_equal(out[0], a)


def test_array_split_chunksize_invalid_chunksize():
    """Assert an error is returned if the chunksize is less than one"""
    with pytest.raises(ValueError, match="chunksize must be greater than 1"):
        array_split_chunksize(np.array([1, 2]), -1)


def test_get_inverse_indices():
    """Assert the correct indices are returned"""
    indices = np.array([1, 2, 3])
    n = 5
    expected = np.array([0, 4])
    out = get_inverse_indices(n, indices)
    np.testing.assert_array_equal(out, expected)


def test_get_inverse_indices_empty_inverse():
    """Assert an empty array is returned in the indices are complete"""
    indices = np.array([0, 1, 2, 3, 4])
    n = 5
    expected = np.array([])
    out = get_inverse_indices(n, indices)
    np.testing.assert_array_equal(out, expected)


def test_get_inverse_indices_invalid_n():
    """Assert"""
    indices = np.array([0, 1, 4])
    n = 4
    with pytest.raises(
        ValueError, match=r"Indices contain values that are out of range for n"
    ):
        get_inverse_indices(n, indices)
