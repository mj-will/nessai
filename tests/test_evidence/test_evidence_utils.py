# -*- coding: utf-8 -*-
"""
Test general utilities used in the evidence computation.
"""
import pytest

from nessai.evidence import (
    logsubexp,
)


def test_logsubexp_negative():
    """
    Test behaviour of logsubexp for x < y
    """
    with pytest.raises(Exception):
        logsubexp(1, 2)
