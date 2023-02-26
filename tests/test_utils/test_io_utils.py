# -*- coding: utf-8 -*-
"""
Test io utilities.
"""
import os
import json
import h5py
import numpy as np
import pickle
import pytest
from unittest.mock import call, create_autospec, mock_open, patch

from nessai import config
from nessai.livepoint import numpy_array_to_live_points
from nessai.utils.io import (
    NessaiJSONEncoder,
    add_dict_to_hdf5_file,
    encode_for_hdf5,
    is_jsonable,
    safe_file_dump,
    save_dict_to_hdf5,
    save_live_points,
    save_to_json,
)


@pytest.fixture
def data_dict():
    data = dict(
        a=np.array([1, 2, 3]),
        b=np.array([(1, 2)], dtype=[("x", "f4"), ("y", "f4")]),
        cls=object(),
        l=[1, 2, 3],
        dict1={"a": None, "b": 2},
        dict2={"c": [1, 2, 3], "array": np.array([3, 4, 5])},
        s="A string",
        nan=None,
    )
    return data


def test_is_jsonable_true():
    """Assert True is return is json.dumps does not raise an error"""
    assert is_jsonable({"x": 2})


@pytest.mark.parametrize("cls", [TypeError, OverflowError])
def test_is_jsonable_false(
    cls,
):
    """Assert True is return is json.dumps does not raise an error"""
    with patch("json.dumps", side_effect=cls()) as mock:
        assert not is_jsonable({"x": 2})
    mock.assert_called_once_with({"x": 2})


@pytest.mark.parametrize(
    "input, expected",
    [(np.int32(2), 2), (np.float64(2), 2.0), (np.array([1, 2]), [1, 2])],
)
def test_JSON_encoder_numpy(input, expected):
    """Test the JSON encoder with numpy inputs"""
    e = create_autospec(NessaiJSONEncoder)
    output = NessaiJSONEncoder.default(e, input)
    assert output == expected


def test_JSON_encoder_object():
    """Test the JSON with an object that is JSONable"""
    e = create_autospec(NessaiJSONEncoder)
    input = object()
    with patch("nessai.utils.io.is_jsonable", return_value=False) as m:
        output = NessaiJSONEncoder.default(e, input)
    m.assert_called_once_with(input)
    assert "object object at" in output


def test_JSON_encoder_other():
    """Test the JSON with an object that is JSONable"""
    e = create_autospec(NessaiJSONEncoder)
    input = "Hello"
    with patch("json.JSONEncoder.default", return_value="Hello") as m:
        output = NessaiJSONEncoder.default(e, input)
    m.assert_called_once_with(input)
    assert output == "Hello"


def test_save_to_json():
    """Assert json.dump is called with the correct arguments"""
    d = dict(a=1)
    expected_kwargs = dict(indent=4, cls=NessaiJSONEncoder, test=True)
    mop = mock_open()
    filename = "test.json"
    with patch("builtins.open", mop, create=True), patch(
        "json.dump"
    ) as mock_dump:
        save_to_json(d, filename, test=True)
    fp = mop()
    mock_dump.assert_called_once_with(d, fp, **expected_kwargs)


@pytest.mark.integration_test
def test_save_to_json_integration(tmp_path, data_dict):
    """Integration test for save to json"""
    filename = tmp_path / "result.json"
    save_to_json(data_dict, filename)
    assert os.path.exists(filename)

    with open(filename, "r") as fp:
        out = json.load(fp)
    assert list(data_dict.keys()) == list(out.keys())


def test_safe_file_dump():
    """Test safe file dump."""
    m = mock_open()
    data = np.array([1, 2])
    with patch("builtins.open", m) as mo, patch("pickle.dump") as md, patch(
        "shutil.move"
    ) as msm:
        safe_file_dump(data, "test.pkl", pickle, save_existing=False)
    mo.assert_called_once_with("test.pkl.temp", "wb")
    md.call_args_list[0][0] == data
    msm.assert_called_once_with("test.pkl.temp", "test.pkl")


def test_safe_file_dump_save_existing():
    """Test safe file dump."""
    import pickle

    m = mock_open()
    data = np.array([1, 2])
    with patch("os.path.exists", return_value=True) as mpe, patch(
        "builtins.open", m
    ) as mo, patch("pickle.dump") as md, patch("shutil.move") as msm:
        safe_file_dump(data, "test.pkl", pickle, save_existing=True)
    mpe.assert_called_once_with("test.pkl")
    mo.assert_called_once_with("test.pkl.temp", "wb")
    md.call_args_list[0][0] == data
    msm.assert_has_calls(
        [
            call("test.pkl", "test.pkl.old"),
            call("test.pkl.temp", "test.pkl"),
        ]
    )


def test_safe_file_dump_integration(tmp_path):
    """Integration test for safe file dump"""
    path = tmp_path
    f = path / "test.pkl"
    f.write_text("a")
    data = "b"
    path = str(path)
    safe_file_dump(
        data, os.path.join(path, "test.pkl"), pickle, save_existing=True
    )
    assert os.path.exists(os.path.join(path, "test.pkl"))
    assert os.path.exists(os.path.join(path, "test.pkl.old"))


def test_save_live_points(tmp_path):
    """Test the function for saving live points"""
    d = {"x": [1, 2], "y": [3, 4]}
    d.update(
        {
            k: 2 * [v]
            for k, v in zip(
                config.livepoints.non_sampling_parameters,
                config.livepoints.non_sampling_defaults,
            )
        }
    )
    live_points = numpy_array_to_live_points(
        np.array([[1, 3], [2, 4]]), ["x", "y"]
    )
    filename = os.path.join(str(tmp_path), "test.json")
    save_live_points(live_points, filename)

    with open(filename, "r") as fp:
        d_out = json.load(fp)

    np.testing.assert_equal(d_out, d)


@pytest.mark.parametrize(
    "value, expected", [(None, "__none__"), ([1, 2], [1, 2])]
)
def test_encode_to_hdf5(value, expected):
    """Assert values are correctly encoded."""
    assert encode_for_hdf5(value) == expected


def test_add_dict_to_hdf5_file(tmp_path, data_dict):
    """Assert dictionaries is correctly converted"""
    data_dict.pop("cls")
    with h5py.File(tmp_path / "test.h5", "w") as f:
        add_dict_to_hdf5_file(f, "/", data_dict)
        assert list(f.keys()) == sorted(data_dict.keys())
        assert f["/dict1/a"][()].decode() == "__none__"
        np.testing.assert_array_equal(
            f["dict2/array"][:], data_dict["dict2"]["array"]
        )


def test_save_dict_to_hdf5(data_dict):
    """Assert the correct arguments are specified"""
    f = mock_open()
    filename = "result.h5"
    with patch("h5py.File", f) as mock_file, patch(
        "nessai.utils.io.add_dict_to_hdf5_file"
    ) as mock_add:
        save_dict_to_hdf5(data_dict, filename)
    mock_file.assert_called_once_with(filename, "w")
    mock_add.assert_called_once_with(f(), "/", data_dict)


@pytest.mark.integration_test
def test_save_dict_to_hdf5_integration(tmp_path, data_dict):
    """Test saving a dict to HDF5 integration"""
    # Do not need to support saving a class to HDF5
    data_dict.pop("cls")
    filename = tmp_path / "result.hdf5"
    save_dict_to_hdf5(data_dict, filename)

    with h5py.File(filename, "r") as f:
        keys = list(f.keys())
    assert keys == sorted(list(data_dict.keys()))
