# -*- coding: utf-8 -*-
"""
Test io utilities.
"""
import pytest
from unittest.mock import patch

from nessai.utils.io import (
    is_jsonable
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
