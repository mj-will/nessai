# -*- coding: utf-8 -*-
"""
Tests for threading related utilities.
"""
import pytest
from unittest.mock import patch

from nessai.utils.threading import configure_threads


def test_configure_threads():
    """Test configuring the threads"""
    with patch("torch.set_num_threads") as mock:
        configure_threads(pytorch_threads=1)
    mock.assert_called_once_with(1)


def test_configure_threads_none():
    """Test configuring the threads.

    Assert `set_num_threads` it not called if pytorch_threads is None.
    """
    with patch("torch.set_num_threads") as mock:
        configure_threads(pytorch_threads=None)
    mock.assert_not_called()


def test_max_threads_warning():
    """Assert a deprecation warning is raised if max threads is specified"""
    with pytest.warns(DeprecationWarning) as record:
        configure_threads(max_threads=1)
    assert "`max_threads` is deprecated" in str(record[0].message)
