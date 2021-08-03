# -*- coding: utf-8 -*-
"""
Tests for utilies related to python structures such as lists.
"""
import numpy as np
import pytest

from nessai.utils.structures import (
    count_matches,
    get_subset_arrays,
    replace_in_list
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
    x = ['aa', 'bb', 'cc']
    replace_in_list(x, ['aa', 'bb'], ['dd', 'ee'])
    assert x == ['dd', 'ee', 'cc']


def test_replace_in_list_single_str():
    """Test the function with single strings"""
    x = ['aa', 'bb', 'cc']
    replace_in_list(x, 'aa', 'dd')
    assert x == ['dd', 'bb', 'cc']


def test_different_lengths():
    """
    Assert an error is raised if the targets and replacements are different
    lengths.
    """
    with pytest.raises(RuntimeError) as excinfo:
        replace_in_list([1, 2], [1, 2], 3)
    assert 'Targets and replacements are different lengths!' \
        in str(excinfo.value)


def test_missing_targets():
    """Assert an error is raised if a target is not in the target list."""
    with pytest.raises(ValueError) as excinfo:
        replace_in_list([1, 2], 4, 3)
    assert 'Targets [4] not in list: [1, 2]' in str(excinfo.value)


def test_count_matches():
    """Assert the correct number of matches is returned"""
    x = np.array([0, 1, 1, 2, 2, 2])
    m = count_matches(x, 1)
    n = count_matches(x, 2)
    assert m == 2
    assert n == 3


def test_get_subset_arrays_indices():
    """Test get subset arrays with indices"""
    indices = np.array([1, 3, 5])
    x = np.arange(6)
    y = -np.arange(6)

    x_out, y_out = get_subset_arrays(indices, x, y)

    np.testing.assert_equal(x_out, [1, 3, 5])
    np.testing.assert_equal(y_out, [-1, -3, -5])


def test_get_subset_arrays_booleans():
    """Test get subset arrays with an array of booleans"""
    valid = np.array([False, True, False, True, False, True])
    x = np.arange(6)
    y = -np.arange(6)

    x_out, y_out = get_subset_arrays(valid, x, y)

    np.testing.assert_equal(x_out, [1, 3, 5])
    np.testing.assert_equal(y_out, [-1, -3, -5])
