
import numpy as np
import pandas as pd
import pytest

from nessai import config
import nessai.livepoint as lp
from nessai.utils.testing import (
    assert_structured_arrays_equal as assert_lp_equal,
)


EXTRA_PARAMS_DTYPE = [
    (nsp, d) for nsp, d in
    zip(config.NON_SAMPLING_PARAMETERS, config.NON_SAMPLING_DEFAULT_DTYPE)
]


@pytest.fixture(autouse=True, params=[[], ['logQ', 'logW']])
def extra_parameters(request):
    """Add (and remove) extra parameters for the tests."""
    # Before every test
    lp.add_extra_parameters_to_live_points(request.param)
    global EXTRA_PARAMS_DTYPE
    EXTRA_PARAMS_DTYPE = [
        (nsp, d) for nsp, d in
        zip(config.NON_SAMPLING_PARAMETERS, config.NON_SAMPLING_DEFAULT_DTYPE)
    ]

    # Test happens here
    yield

    # Called after every test
    lp.reset_extra_live_points_parameters()
    EXTRA_PARAMS_DTYPE = [
        (nsp, d) for nsp, d in
        zip(config.NON_SAMPLING_PARAMETERS, config.NON_SAMPLING_DEFAULT_DTYPE)
    ]


@pytest.fixture
def live_point():
    return np.array(
        [(1., 2., 3., *config.NON_SAMPLING_DEFAULTS)],
        dtype=[
            ('x', config.DEFAULT_FLOAT_DTYPE),
            ('y', config.DEFAULT_FLOAT_DTYPE),
            ('z', config.DEFAULT_FLOAT_DTYPE)
        ] + EXTRA_PARAMS_DTYPE
    )


@pytest.fixture
def live_points():
    return np.array(
        [(1., 2., 3., *config.NON_SAMPLING_DEFAULTS),
         (4., 5., 6., *config.NON_SAMPLING_DEFAULTS)],
        dtype=[
            ('x', config.DEFAULT_FLOAT_DTYPE),
            ('y', config.DEFAULT_FLOAT_DTYPE),
            ('z', config.DEFAULT_FLOAT_DTYPE)
        ] + EXTRA_PARAMS_DTYPE
    )


@pytest.fixture
def empty_live_point():
    return np.empty(
        0,
        dtype=[
            ('x', config.DEFAULT_FLOAT_DTYPE),
            ('y', config.DEFAULT_FLOAT_DTYPE),
            ('z', config.DEFAULT_FLOAT_DTYPE),
        ] + EXTRA_PARAMS_DTYPE
    )


def test_get_dtype():
    """Assert the correct value is returned"""
    names = ['x', 'y']
    expected = \
        [('x', 'f4'), ('y', 'f4')] \
        + [('logP', config.DEFAULT_FLOAT_DTYPE), ('logL', config.LOGL_DTYPE),
           ('it', config.IT_DTYPE)] \
        + list(zip(config.EXTRA_PARAMETERS, config.EXTRA_PARAMETERS_DTYPE))
    dtype = lp.get_dtype(names, array_dtype='f4')
    assert dtype == expected


def test_empty_structured_array_names():
    """Assert the correct default values are used when specifying names"""
    n = 10
    fields = ['x', 'y']
    array = lp.empty_structured_array(n, names=fields)
    for f in fields:
        np.testing.assert_array_equal(
            array[f], config.DEFAULT_FLOAT_VALUE * np.ones(n)
        )
    for nsp, v in zip(
        config.NON_SAMPLING_PARAMETERS, config.NON_SAMPLING_DEFAULTS
    ):
        np.testing.assert_array_equal(array[nsp], v * np.ones(n))


def test_empty_structured_array_dtype():
    """Assert the correct default values are used when specifying the dtype"""
    n = 10
    dtype = [('x', 'f8'),  ('y', 'f8')] + EXTRA_PARAMS_DTYPE
    array = lp.empty_structured_array(n, dtype=dtype)
    for f in ['x', 'y']:
        np.testing.assert_array_equal(
            array[f], config.DEFAULT_FLOAT_VALUE * np.ones(n)
        )
    assert array.dtype == dtype


def test_empty_structured_array_zero_points():
    """Assert setting n=0 works correctly"""
    dtype = [('x', 'f8'),  ('y', 'f8')] + EXTRA_PARAMS_DTYPE
    array = lp.empty_structured_array(0, names=['x', 'y'])
    assert len(array) == 0
    assert array.dtype == dtype


def test_empty_structured_array_dtype_missing():
    """Assert an error is raised if the non-sampling parameters are missing"""
    n = 10
    dtype = [('x', 'f8'),  ('y', 'f8')]
    with pytest.raises(ValueError) as excinfo:
        lp.empty_structured_array(n, dtype=dtype)
    assert "non-sampling parameters" in str(excinfo.value)


def test_parameters_to_live_point(live_point):
    """
    Test function that produces a single live point given the parameter
    values for the live point as a live or an array
    """
    x = lp.parameters_to_live_point([1., 2., 3.], ['x', 'y', 'z'])
    assert_lp_equal(x, live_point)


def test_empty_parameters_to_live_point(empty_live_point):
    """
    Test behaviour when an empty parameter is parsed
    """
    np.testing.assert_array_equal(
        lp.parameters_to_live_point([], ['x', 'y', 'z']),
        empty_live_point,
    )


def test_numpy_array_to_live_point(live_point):
    """
    Test the function the produces an array of live points given numpy array
    of shape [# dimensions]
    """
    array = np.array([1., 2., 3.])
    x = lp.numpy_array_to_live_points(array, names=['x', 'y', 'z'])
    assert_lp_equal(live_point, x)


def test_numpy_array_multiple_to_live_points(live_points):
    """
    Test the function the produces an array of live points given numpy array
    of shape [# point, # dimensions]
    """
    array = np.array([[1., 2., 3.], [4., 5., 6.]])
    x = lp.numpy_array_to_live_points(array, names=['x', 'y', 'z'])
    assert_lp_equal(live_points, x)


def test_empty_numpy_array_to_live_points(empty_live_point):
    """
    Test the function the produces an array of live points given an empty
    numpy array
    """
    np.testing.assert_array_equal(
        empty_live_point,
        lp.numpy_array_to_live_points(np.array([]), names=['x', 'y', 'z'])
    )


@pytest.mark.parametrize(
    'd',
    [
        {'x': 1, 'y': 2, 'z': 3},
        {'x': 1.0, 'y': 2.0, 'z': 3.0},
    ]
)
def test_dict_to_live_point(live_point, d):
    """
    Test the function that converts a dictionary with a single live point to
    a live point array
    """
    x = lp.dict_to_live_points(d)
    assert_lp_equal(live_point, x)


@pytest.mark.parametrize(
    'd',
    [
        {'x': [1, 4], 'y': [2, 5], 'z': [3, 6]},
        {'x': np.array([1, 4]), 'y': np.array([2, 5]), 'z': np.array([3, 6])},
    ]
)
def test_dict_multiple_to_live_points(live_points, d):
    """
    Test the function that converts a dictionary with multiple live points to
    a live point array
    """
    x = lp.dict_to_live_points(d)
    assert_lp_equal(live_points, x)


def test_empty_dict_to_live_points(empty_live_point):
    """
    Test the function that converts a dictionary with empty lists to
    a live point array
    """
    np.testing.assert_array_equal(
        empty_live_point,
        lp.dict_to_live_points({'x': [], 'y': [], 'z': []})
    )


def test_dataframe_to_live_points(live_points):
    """Test converting from a pandas dataframe to live points."""
    df = pd.DataFrame({'x': [1, 4], 'y': [2, 5], 'z': [3, 6]})
    out = lp.dataframe_to_live_points(df)
    assert_lp_equal(out, live_points)


def test_live_point_to_numpy_array(live_point):
    """
    Test conversion from a live point to an unstructured numpy array
    """
    (
        np.array([[1, 2, 3, *config.NON_SAMPLING_DEFAULTS]]),
        lp.live_points_to_array(live_point)
    )


def test_live_point_to_numpy_array_with_names(live_point):
    """
    Test conversion from a live point to an unstructured numpy array with
    specific fields
    """
    np.testing.assert_array_equal(
        np.array([[1, 3, np.nan]]),
        lp.live_points_to_array(live_point, names=['x', 'z', 'logP'])
    )


def test_live_point_to_dict(live_point):
    """
    Test conversion of a live point to a dictionary
    """
    d = {'x': 1., 'y': 2., 'z': 3.}
    d.update(
        {k: v for k, v in
         zip(config.NON_SAMPLING_PARAMETERS, config.NON_SAMPLING_DEFAULTS)}
    )
    np.testing.assert_equal(lp.live_points_to_dict(live_point), d)


def test_live_point_to_dict_with_names(live_point):
    """
    Test conversion of a live point to a dictionary
    """
    d = {'x': np.array([1.]), 'z': np.array([3.]), 'logP': np.array([np.nan])}
    np.testing.assert_equal(
        lp.live_points_to_dict(live_point, names=['x', 'z', 'logP']), d
    )


def test_multiple_live_points_to_dict(live_points):
    """
    Test conversion of multiple_live points to a dictionary
    """
    d = {'x': [1, 4], 'y': [2, 5], 'z': [3, 6]}
    d.update({
        k: 2 * [v] for k, v in
        zip(config.NON_SAMPLING_PARAMETERS, config.NON_SAMPLING_DEFAULTS)
    })
    d_out = lp.live_points_to_dict(live_points)
    assert list(d.keys()) == list(d_out.keys())
    np.testing.assert_array_equal(list(d.values()), list(d_out.values()))
