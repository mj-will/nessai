# -*- coding: utf-8 -*-
"""
Tests for torch utils.
"""
from unittest.mock import patch

import pytest
import torch

from nessai.utils.torchutils import set_torch_default_dtype


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("float32", torch.float32),
        ("float64", torch.float64),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_setting_default_dtype(dtype, expected):
    """Assert `set_default_dtype` is called correctly"""
    with patch("torch.set_default_dtype") as mock:
        out = set_torch_default_dtype(dtype)
    mock.assert_called_once_with(expected)
    assert out is expected


def test_setting_default_none():
    """Assert the default dtype is used when dtype is None"""
    with patch("torch.get_default_dtype", return_value=torch.float32) as mock:
        out = set_torch_default_dtype(None)
    mock.assert_called_once()
    assert out is torch.float32


def test_setting_default_dtype_invalid_input():
    """Assert an error is raised if the input is invalid"""
    with pytest.raises(ValueError) as excinfo:
        set_torch_default_dtype("not_a_dtype")
    assert "Unknown torch dtype: not_a_dtype" in str(excinfo.value)
