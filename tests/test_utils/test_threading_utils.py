# -*- coding: utf-8 -*-
"""
Tests for threading related utilities.
"""
import pytest
from unittest.mock import patch

from nessai.utils.threading import configure_threads


@pytest.mark.parametrize(
    'max_threads, pytorch_threads, n_pool, expected',
    [(2, 1, 1, 1), (1, None, None, 1), (2, None, 1, 1), (None, None, 4, None)]
)
def test_configure_threads(
    max_threads, pytorch_threads, n_pool, expected
):
    """Test configuring the threads"""
    with patch('torch.set_num_threads') as mock:
        configure_threads(
            max_threads=max_threads,
            pytorch_threads=pytorch_threads,
            n_pool=n_pool
        )
    if expected:
        mock.assert_called_once_with(expected)
    else:
        mock.assert_not_called()


def test_configure_threads_pytorch_error():
    """Assert an error is raised if pytorch_threads > max_threads"""
    with pytest.raises(RuntimeError) as excinfo:
        configure_threads(max_threads=2, pytorch_threads=3)
    assert 'More threads assigned to PyTorch (3) than' in str(excinfo.value)


def test_configure_threads_n_pool_error():
    """Assert an error is raised if n_pool >= max_threads"""
    with pytest.raises(RuntimeError) as excinfo:
        configure_threads(max_threads=2, n_pool=2)
    assert 'More threads assigned to pool (2) than' in str(excinfo.value)


def test_configure_threads_both_error():
    """Assert an error is raised if pytorch_threads + n_pool > max_threads"""
    with pytest.raises(RuntimeError) as excinfo:
        configure_threads(max_threads=3, pytorch_threads=2, n_pool=2)
    assert 'More threads assigned to PyTorch (2) and pool (2)' \
        in str(excinfo.value)
