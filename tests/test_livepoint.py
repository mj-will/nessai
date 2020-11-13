
import numpy as np
import pytest

import nessai.livepoint as lp


@pytest.fixture
def live_point():
    return np.array([(1., 2., 3., 0., 0.)],
                    dtype=[('x', lp.DEFAULT_FLOAT_DTYPE),
                           ('y', lp.DEFAULT_FLOAT_DTYPE),
                           ('z', lp.DEFAULT_FLOAT_DTYPE),
                           ('logP', lp.DEFAULT_FLOAT_DTYPE),
                           ('logL', lp.LOGL_DTYPE)])


@pytest.fixture
def live_points():
    return np.array([(1., 2., 3., 0., 0.),
                     (4., 5., 6., 0., 0.)],
                    dtype=[('x', lp.DEFAULT_FLOAT_DTYPE),
                           ('y', lp.DEFAULT_FLOAT_DTYPE),
                           ('z', lp.DEFAULT_FLOAT_DTYPE),
                           ('logP', lp.DEFAULT_FLOAT_DTYPE),
                           ('logL', lp.LOGL_DTYPE)])


@pytest.fixture
def empty_live_point():
    return np.empty(0, dtype=[('x', lp.DEFAULT_FLOAT_DTYPE),
                              ('y', lp.DEFAULT_FLOAT_DTYPE),
                              ('z', lp.DEFAULT_FLOAT_DTYPE),
                              ('logP', lp.DEFAULT_FLOAT_DTYPE),
                              ('logL', lp.LOGL_DTYPE)])


def test_parameters_to_live_point(live_point):
    """
    Test function that produces a single live point given the parameter
    values for the live point as a live or an array
    """
    x = lp.parameters_to_live_point([1., 2., 3.], ['x', 'y', 'z'])
    np.testing.assert_array_equal(live_point, x)


def test_empty_parameters_to_live_point(empty_live_point):
    """
    Test behaviour when an empty parameter is parsed
    """
    np.testing.assert_array_equal(
        empty_live_point,
        lp.parameters_to_live_point([], ['x', 'y', 'z']))


def test_numpy_array_to_live_point(live_point):
    """
    Test the fuction the produces an array of live points given numpy array
    of shape [# dimensions]
    """
    array = np.array([1., 2., 3.])
    x = lp.numpy_array_to_live_points(array, names=['x', 'y', 'z'])
    np.testing.assert_array_equal(live_point, x)


def test_numpy_array_multiple_to_live_points(live_points):
    """
    Test the fuction the produces an array of live points given numpy array
    of shape [# point, # dimensions]
    """
    array = np.array([[1., 2., 3.], [4., 5., 6.]])
    x = lp.numpy_array_to_live_points(array, names=['x', 'y', 'z'])
    np.testing.assert_array_equal(live_points, x)


def test_empty_numpy_array_to_live_points(empty_live_point):
    """
    Test the fuction the produces an array of live points given an empty
    numpy array
    """
    np.testing.assert_array_equal(
        empty_live_point,
        lp.numpy_array_to_live_points(np.array([]), names=['x', 'y', 'z']))


def test_dict_to_live_point(live_point):
    """
    Test the function that converts a dictionary with a signel live point to
    a live point array
    """
    d = {'x': 1, 'y': 2, 'z': 3}
    x = lp.dict_to_live_points(d)
    np.testing.assert_array_equal(live_point, x)


def test_dict_multiple_to_live_points(live_points):
    """
    Test the function that converts a dictionary with multiple live points to
    a live point array
    """
    d = {'x': [1, 4], 'y': [2, 5], 'z': [3, 6]}
    x = lp.dict_to_live_points(d)
    np.testing.assert_array_equal(live_points, x)


def test_empty_dict_to_live_points(empty_live_point):
    """
    Test the function that converts a dictionary with empty lists to
    a live point array
    """
    np.testing.assert_array_equal(
        empty_live_point,
        lp.dict_to_live_points({'x': [], 'y': [], 'z': []}))


def test_live_point_to_numpy_array(live_point):
    """
    Test coversion from a live point to an unstructured numpy array
    """
    np.testing.assert_array_equal(
        np.array([[1, 2, 3, 0, 0]]),
        lp.live_points_to_array(live_point))


def test_live_point_to_numpy_array_with_names(live_point):
    """
    Test coversion from a live point to an unstructured numpy array with
    specific fields
    """
    np.testing.assert_array_equal(
        np.array([[1, 3, 0]]),
        lp.live_points_to_array(live_point, names=['x', 'z', 'logP']))


def test_live_point_to_dict(live_point):
    """
    Test conversion of a live point to a dictionary
    """
    d = {'x': 1., 'y': 2., 'z': 3., 'logP': 0., 'logL': 0.}
    assert d == lp.live_points_to_dict(live_point)


def test_live_point_to_dict_with_names(live_point):
    """
    Test conversion of a live point to a dictionary
    """
    d = {'x': 1., 'z': 3., 'logP': 0.}
    assert d == lp.live_points_to_dict(live_point, names=['x', 'z', 'logP'])


def test_multiple_live_points_to_dict(live_points):
    """
    Test conversion of multiple_live points to a dictionary
    """
    d = {'x': [1, 4], 'y': [2, 5], 'z': [3, 6], 'logP': [0, 0], 'logL': [0, 0]}
    d_out = lp.live_points_to_dict(live_points)
    assert list(d.keys()) == list(d_out.keys())
    np.testing.assert_array_equal(list(d.values()), list(d_out.values()))
