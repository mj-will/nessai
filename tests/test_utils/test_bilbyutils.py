# -*- coding: utf-8 -*-
"""
Tests for bilbyutils
"""
from unittest.mock import patch
from nessai.flowsampler import FlowSampler
from nessai.utils.bilbyutils import (
    get_all_kwargs,
    get_run_kwargs_list,
    _get_standard_methods,
)


def test_get_standard_methods():
    """Assert a list of methods is returned"""
    out = _get_standard_methods()
    assert len(out) == 5


def test_get_all_kwargs():
    """Assert the correct dictionary is returned.

    Positional arguments should be ignored.
    """

    def func0(a, b, c=2, d=None):
        pass

    def func1(e, f, g=3, h=True):
        pass

    expected = dict(c=2, d=None, g=3, h=True)

    with patch(
        "nessai.utils.bilbyutils._get_standard_methods",
        return_value=[func0, func1],
    ):
        out = get_all_kwargs()

    assert out == expected


def test_get_run_kwargs_list():
    """Assert the correct list is returned"""

    def func(a, b=1, c=None):
        pass

    expected = ["b", "c"]

    with patch.object(FlowSampler, "run", func):
        out = get_run_kwargs_list()

    assert out == expected
