# -*- coding: utf-8 -*-
"""
Test io utilities.
"""
import os
import json
import numpy as np
import pickle
import pytest
from unittest.mock import call, create_autospec, mock_open, patch

from nessai.livepoint import numpy_array_to_live_points
from nessai.utils.io import (
    NessaiJSONEncoder,
    is_jsonable,
    safe_file_dump,
    save_live_points
)


def test_is_jsonable_true():
    """Assert True is return is json.dumps does not raise an error"""
    assert is_jsonable({'x': 2})


@pytest.mark.parametrize('cls', [TypeError, OverflowError])
def test_is_jsonable_false(cls, ):
    """Assert True is return is json.dumps does not raise an error"""
    with patch('json.dumps', side_effect=cls()) as mock:
        assert not is_jsonable({'x': 2})
    mock.assert_called_once_with({'x': 2})


@pytest.mark.parametrize(
    'input, expected',
    [(np.int32(2), 2), (np.float64(2), 2.0), (np.array([1, 2]), [1, 2])]
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
    with patch('nessai.utils.io.is_jsonable', return_value=False) as m:
        output = NessaiJSONEncoder.default(e, input)
    m.assert_called_once_with(input)
    assert 'object object at' in output


def test_JSON_encoder_other():
    """Test the JSON with an object that is JSONable"""
    e = create_autospec(NessaiJSONEncoder)
    input = 'Hello'
    with patch('json.JSONEncoder.default', return_value='Hello') as m:
        output = NessaiJSONEncoder.default(e, input)
    m.assert_called_once_with(input)
    assert output == 'Hello'


def test_safe_file_dump():
    """Test safe file dump."""
    m = mock_open()
    data = np.array([1, 2])
    with patch('builtins.open', m) as mo,\
         patch('pickle.dump') as md,\
         patch('shutil.move') as msm:
        safe_file_dump(data, 'test.pkl', pickle, save_existing=False)
    mo.assert_called_once_with('test.pkl.temp', 'wb')
    md.call_args_list[0][0] == data
    msm.assert_called_once_with('test.pkl.temp', 'test.pkl')


def test_safe_file_dump_save_existing():
    """Test safe file dump."""
    import pickle
    m = mock_open()
    data = np.array([1, 2])
    with patch('os.path.exists', return_value=True) as mpe,\
         patch('builtins.open', m) as mo,\
         patch('pickle.dump') as md,\
         patch('shutil.move') as msm:
        safe_file_dump(data, 'test.pkl', pickle, save_existing=True)
    mpe.assert_called_once_with('test.pkl')
    mo.assert_called_once_with('test.pkl.temp', 'wb')
    md.call_args_list[0][0] == data
    msm.assert_has_calls([
        call('test.pkl', 'test.pkl.old'), call('test.pkl.temp', 'test.pkl'),
    ])


def test_safe_file_dump_integration(tmp_path):
    """Integration test for safe file dump"""
    path = tmp_path
    f = path / 'test.pkl'
    f.write_text('a')
    data = 'b'
    safe_file_dump(data, str(path) + '/test.pkl', pickle, save_existing=True)
    assert os.path.exists(str(path) + '/test.pkl')
    assert os.path.exists(str(path) + '/test.pkl.old')


def test_save_live_points(tmp_path):
    """Test the function for saving live points"""
    d = {'x': [1, 2], 'y': [3, 4], 'logP': [0, 0], 'logL': [0, 0]}
    live_points = numpy_array_to_live_points(
        np.array([[1, 3], [2, 4]]), ['x', 'y']
    )
    filename = str(tmp_path) + 'test.json'
    save_live_points(live_points, filename)

    with open(filename, 'r') as fp:
        d_out = json.load(fp)

    assert d_out == d
