# -*- coding: utf-8 -*-
import pytest
import numpy as np

from nessai.posterior import (
    logsubexp,
    LogNegativeError
)


def test_logsubexp():
    """Test the values returned by logsubexp"""
    out = logsubexp(2, 1)
    np.testing.assert_almost_equal(out, np.log(np.exp(2) - np.exp(1)),
                                   decimal=12)


def test_logsubexp_negative():
    """
    Test behaviour of logsubexp for x < y
    """
    with pytest.raises(LogNegativeError):
        logsubexp(1, 2)
