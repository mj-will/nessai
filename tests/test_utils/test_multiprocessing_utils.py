# -*- coding: utf-8 -*-
"""
Tests for rescaling functions
"""
from multiprocessing.dummy import Pool
import pytest
from unittest.mock import MagicMock

from nessai.utils.multiprocessing import (
    initialise_pool_variables,
    log_likelihood_wrapper,
)


def test_pool_variables():
    """Assert initialising the variables and calling the likelihood work"""
    model = MagicMock()
    model.log_likelihood = lambda x: x
    initialise_pool_variables(model)
    pool = Pool(1)
    out = pool.map(log_likelihood_wrapper, [1, 2, 3])
    pool.close()
    pool.terminate()
    assert out == [1, 2, 3]

    # Reset to the default value
    initialise_pool_variables(None)


def test_model_error():
    """Assert an error is raised in the global variables have not been \
        initialised
    """
    model = MagicMock()
    model.log_likelihood = lambda x: x
    pool = Pool(1)
    with pytest.raises(AttributeError) as excinfo:
        pool.map(log_likelihood_wrapper, [1, 2, 3])
    assert "'NoneType' object has no attribute 'log_likelihood'" \
        in str(excinfo.value)
    pool.close()
    pool.terminate()
