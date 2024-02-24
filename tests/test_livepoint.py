"""Tests for livepoint functions"""

import sys

import numpy as np
import pandas as pd
import pytest

from nessai import config
import nessai.livepoint as lp
from nessai.utils.testing import assert_structured_arrays_equal


EXTRA_PARAMS_DTYPE = [
    (nsp, d)
    for nsp, d in zip(
        config.livepoints.non_sampling_parameters,
        config.livepoints.non_sampling_dtype,
    )
]


def assert_lp_equal(x, y, non_sampling_parameters=True):
    """Custom assertion that can skip the non-sampling parameters"""
    if non_sampling_parameters:
        assert_structured_arrays_equal(x, y)
    else:
        for nsp in config.livepoints.non_sampling_parameters:
            if nsp in x.dtype.names:
                raise AttributeError(
                    "x has non-sampling parameters but "
                    f"non_sampling_parameters=False. x dtype: {x.dtype}"
                )

        names = [
            n
            for n in x.dtype.names
            if n not in config.livepoints.non_sampling_parameters
        ]
        y = np.lib.recfunctions.repack_fields(y[names])
        assert_structured_arrays_equal(x, y)


@pytest.fixture(params=[True, False])
def non_sampling_parameters(request):
    return request.param


@pytest.fixture(autouse=True, params=[[], ["logQ", "logW"]])
def extra_parameters(request):
    """Add (and remove) extra parameters for the tests."""
    # Before every test
    lp.reset_extra_live_points_parameters()
    lp.add_extra_parameters_to_live_points(request.param)
    global EXTRA_PARAMS_DTYPE
    EXTRA_PARAMS_DTYPE = [
        (nsp, d)
        for nsp, d in zip(
            config.livepoints.non_sampling_parameters,
            config.livepoints.non_sampling_dtype,
        )
    ]

    # Test happens here
    yield

    # Called after every test
    lp.reset_extra_live_points_parameters()
    EXTRA_PARAMS_DTYPE = [
        (nsp, d)
        for nsp, d in zip(
            config.livepoints.non_sampling_parameters,
            config.livepoints.non_sampling_dtype,
        )
    ]


@pytest.fixture(params=["f4", "f16"])
def change_dtype(request):
    """Fixture that changes the default float dtype"""
    dtype = request.param
    if dtype == "f16" and sys.platform.startswith("win"):
        pytest.skip("Skipping test with float128 on Windows.")
    current_dtype = config.livepoints.default_float_dtype
    config.livepoints.default_float_dtype = dtype

    yield dtype

    config.livepoints.default_float_dtype = current_dtype


@pytest.fixture
def live_point():
    return np.array(
        [(1.0, 2.0, 3.0, *config.livepoints.non_sampling_defaults)],
        dtype=[
            ("x", config.livepoints.default_float_dtype),
            ("y", config.livepoints.default_float_dtype),
            ("z", config.livepoints.default_float_dtype),
        ]
        + EXTRA_PARAMS_DTYPE,
    )


@pytest.fixture
def live_points():
    return np.array(
        [
            (1.0, 2.0, 3.0, *config.livepoints.non_sampling_defaults),
            (4.0, 5.0, 6.0, *config.livepoints.non_sampling_defaults),
        ],
        dtype=[
            ("x", config.livepoints.default_float_dtype),
            ("y", config.livepoints.default_float_dtype),
            ("z", config.livepoints.default_float_dtype),
        ]
        + EXTRA_PARAMS_DTYPE,
    )


@pytest.fixture
def empty_live_point():
    return np.empty(
        0,
        dtype=[
            ("x", config.livepoints.default_float_dtype),
            ("y", config.livepoints.default_float_dtype),
            ("z", config.livepoints.default_float_dtype),
        ]
        + EXTRA_PARAMS_DTYPE,
    )


def test_add_extra_parameters(caplog):
    """Assert a warning is raised when adding a parameter name twice."""
    lp.add_extra_parameters_to_live_points(["test"], [1.0])
    assert "test" in config.livepoints.extra_parameters
    loc = config.livepoints.extra_parameters.index("test")
    assert config.livepoints.extra_parameters_defaults[loc] == 1.0
    lp.add_extra_parameters_to_live_points(["test"], [1.0])
    assert "Extra parameter `test` has already been added" in str(caplog.text)


def test_get_dtype():
    """Assert the correct value is returned"""
    names = ["x", "y"]
    expected = (
        [("x", "f4"), ("y", "f4")]
        + [
            ("logP", config.livepoints.default_float_dtype),
            ("logL", config.livepoints.logl_dtype),
            ("it", config.livepoints.it_dtype),
        ]
        + list(
            zip(
                config.livepoints.extra_parameters,
                config.livepoints.extra_parameters_dtype,
            )
        )
    )
    dtype = lp.get_dtype(names, array_dtype="f4")
    assert dtype == expected


def test_get_dtype_change_dtype(change_dtype):
    """Assert the dtype can be changed"""
    dtype = lp.get_dtype(["x"])
    assert dtype.fields["x"][0] == np.dtype(change_dtype)


def test_empty_structured_array_names(non_sampling_parameters):
    """Assert the correct default values are used when specifying names"""
    n = 10
    fields = ["x", "y"]
    array = lp.empty_structured_array(
        n, names=fields, non_sampling_parameters=non_sampling_parameters
    )
    for f in fields:
        np.testing.assert_array_equal(
            array[f], config.livepoints.default_float_value * np.ones(n)
        )
    for nsp, v in zip(
        config.livepoints.non_sampling_parameters,
        config.livepoints.non_sampling_defaults,
    ):
        if non_sampling_parameters:
            np.testing.assert_array_equal(array[nsp], v * np.ones(n))
        else:
            assert nsp not in array.dtype.names


def test_empty_structured_array_dtype(non_sampling_parameters):
    """Assert the correct default values are used when specifying the dtype"""
    n = 10
    dtype = [("x", "f8"), ("y", "f8")]
    if non_sampling_parameters:
        dtype += EXTRA_PARAMS_DTYPE
    array = lp.empty_structured_array(
        n, dtype=dtype, non_sampling_parameters=non_sampling_parameters
    )
    for f in ["x", "y"]:
        np.testing.assert_array_equal(
            array[f], config.livepoints.default_float_value * np.ones(n)
        )
    assert array.dtype == dtype


def test_empty_structured_array_zero_points(non_sampling_parameters):
    """Assert setting n=0 works correctly"""
    dtype = [("x", "f8"), ("y", "f8")]
    if non_sampling_parameters:
        dtype += EXTRA_PARAMS_DTYPE
    array = lp.empty_structured_array(
        0, names=["x", "y"], non_sampling_parameters=non_sampling_parameters
    )
    assert len(array) == 0
    assert array.dtype == dtype


def test_empty_structured_array_dtype_missing():
    """Assert an error is raised if the non-sampling parameters are missing"""
    n = 10
    dtype = [("x", "f8"), ("y", "f8")]
    with pytest.raises(ValueError) as excinfo:
        lp.empty_structured_array(n, dtype=dtype)
    assert "non-sampling parameters" in str(excinfo.value)


def test_empty_structured_array_change_dtype(change_dtype):
    """Assert changing the default dtype changes the output"""
    array = lp.empty_structured_array(1, names=["x"])
    assert array.dtype.fields["x"][0] == np.dtype(change_dtype)


def test_parameters_to_live_point(live_point, non_sampling_parameters):
    """
    Test function that produces a single live point given the parameter
    values for the live point as a live or an array
    """
    x = lp.parameters_to_live_point(
        [1.0, 2.0, 3.0],
        ["x", "y", "z"],
        non_sampling_parameters=non_sampling_parameters,
    )
    assert_lp_equal(
        x, live_point, non_sampling_parameters=non_sampling_parameters
    )


def test_empty_parameters_to_live_point(empty_live_point):
    """
    Test behaviour when an empty parameter is parsed
    """
    np.testing.assert_array_equal(
        lp.parameters_to_live_point([], ["x", "y", "z"]),
        empty_live_point,
    )


def test_numpy_array_to_live_point(live_point, non_sampling_parameters):
    """
    Test the function the produces an array of live points given numpy array
    of shape [# dimensions]
    """
    array = np.array([1.0, 2.0, 3.0])
    x = lp.numpy_array_to_live_points(
        array,
        names=["x", "y", "z"],
        non_sampling_parameters=non_sampling_parameters,
    )
    assert_lp_equal(
        x, live_point, non_sampling_parameters=non_sampling_parameters
    )


def test_numpy_array_multiple_to_live_points(
    live_points, non_sampling_parameters
):
    """
    Test the function the produces an array of live points given numpy array
    of shape [# point, # dimensions]
    """
    array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = lp.numpy_array_to_live_points(
        array,
        names=["x", "y", "z"],
        non_sampling_parameters=non_sampling_parameters,
    )
    assert_lp_equal(
        x, live_points, non_sampling_parameters=non_sampling_parameters
    )


def test_empty_numpy_array_to_live_points(empty_live_point):
    """
    Test the function the produces an array of live points given an empty
    numpy array
    """
    np.testing.assert_array_equal(
        empty_live_point,
        lp.numpy_array_to_live_points(np.array([]), names=["x", "y", "z"]),
    )


@pytest.mark.parametrize(
    "d",
    [
        {"x": 1, "y": 2, "z": 3},
        {"x": 1.0, "y": 2.0, "z": 3.0},
    ],
)
def test_dict_to_live_point(live_point, d, non_sampling_parameters):
    """
    Test the function that converts a dictionary with a single live point to
    a live point array
    """
    x = lp.dict_to_live_points(
        d, non_sampling_parameters=non_sampling_parameters
    )
    assert_lp_equal(
        x, live_point, non_sampling_parameters=non_sampling_parameters
    )


@pytest.mark.parametrize(
    "d",
    [
        {"x": [1, 4], "y": [2, 5], "z": [3, 6]},
        {"x": np.array([1, 4]), "y": np.array([2, 5]), "z": np.array([3, 6])},
    ],
)
def test_dict_multiple_to_live_points(live_points, d, non_sampling_parameters):
    """
    Test the function that converts a dictionary with multiple live points to
    a live point array
    """
    x = lp.dict_to_live_points(
        d, non_sampling_parameters=non_sampling_parameters
    )
    assert_lp_equal(
        x, live_points, non_sampling_parameters=non_sampling_parameters
    )


def test_empty_dict_to_live_points(empty_live_point):
    """
    Test the function that converts a dictionary with empty lists to
    a live point array
    """
    np.testing.assert_array_equal(
        empty_live_point, lp.dict_to_live_points({"x": [], "y": [], "z": []})
    )


def test_dataframe_to_live_points(live_points, non_sampling_parameters):
    """Test converting from a pandas dataframe to live points."""
    df = pd.DataFrame({"x": [1, 4], "y": [2, 5], "z": [3, 6]})
    out = lp.dataframe_to_live_points(
        df, non_sampling_parameters=non_sampling_parameters
    )
    assert_lp_equal(out, live_points, non_sampling_parameters)


def test_live_point_to_numpy_array(live_point):
    """
    Test conversion from a live point to an unstructured numpy array
    """
    (
        np.array([[1, 2, 3, *config.livepoints.non_sampling_defaults]]),
        lp.live_points_to_array(live_point),
    )


def test_live_point_to_numpy_array_with_names(live_point):
    """
    Test conversion from a live point to an unstructured numpy array with
    specific fields
    """
    np.testing.assert_array_equal(
        np.array([[1, 3, np.nan]]),
        lp.live_points_to_array(live_point, names=["x", "z", "logP"]),
    )


def test_live_point_to_dict(live_point):
    """
    Test conversion of a live point to a dictionary
    """
    d = {"x": 1.0, "y": 2.0, "z": 3.0}
    d.update(
        {
            k: v
            for k, v in zip(
                config.livepoints.non_sampling_parameters,
                config.livepoints.non_sampling_defaults,
            )
        }
    )
    np.testing.assert_equal(lp.live_points_to_dict(live_point), d)


def test_live_point_to_dict_with_names(live_point):
    """
    Test conversion of a live point to a dictionary
    """
    d = {
        "x": np.array([1.0]),
        "z": np.array([3.0]),
        "logP": np.array([np.nan]),
    }
    np.testing.assert_equal(
        lp.live_points_to_dict(live_point, names=["x", "z", "logP"]), d
    )


def test_multiple_live_points_to_dict(live_points):
    """
    Test conversion of multiple_live points to a dictionary
    """
    d = {"x": [1, 4], "y": [2, 5], "z": [3, 6]}
    d.update(
        {
            k: 2 * [v]
            for k, v in zip(
                config.livepoints.non_sampling_parameters,
                config.livepoints.non_sampling_defaults,
            )
        }
    )
    d_out = lp.live_points_to_dict(live_points)
    assert list(d.keys()) == list(d_out.keys())
    np.testing.assert_array_equal(list(d.values()), list(d_out.values()))


def test_unstructured_view_dtype(live_points):
    """Assert the correct array is returned when given the dtype"""
    dtype = np.dtype({n: live_points.dtype.fields[n] for n in ["x", "y"]})
    view = lp.unstructured_view(live_points, dtype=dtype)
    assert view.base is live_points
    assert view.shape == (live_points.size, 2)


def test_unstructured_view_names(live_points):
    """Assert the correct array is returned when given names"""
    view = lp.unstructured_view(live_points, names=["x", "y"])
    assert view.base is live_points
    assert view.shape == (live_points.size, 2)


def test_unstructured_view_error(live_points):
    """Assert an error is raised when neither names or dtype is given."""
    with pytest.raises(TypeError):
        lp.unstructured_view(live_points)
