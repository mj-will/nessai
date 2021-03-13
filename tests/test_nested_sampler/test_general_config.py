# -*- coding: utf-8 -*-
"""
Test general config functions called when the nested sampler is initialised.
"""
from unittest.mock import patch


@patch('numpy.random.seed')
@patch('torch.manual_seed')
def test_set_random_seed(mock1, mock2, sampler):
    """Test the correct functions are called when setting the random seed"""
    sampler.setup_random_seed(150914)
    mock1.assert_called_once_with(150914)
    mock2.assert_called_once_with(seed=150914)


@patch('numpy.random.seed')
@patch('torch.manual_seed')
def test_no_random_seed(mock1, mock2, sampler):
    """Assert no seed is set if seed=None"""
    sampler.setup_random_seed(None)
    mock1.assert_not_called()
    mock2.assert_not_called()
