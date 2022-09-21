# -*- coding: utf-8 -*-
"""
Tests for threading related utilities.
"""
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
