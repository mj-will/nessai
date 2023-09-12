# -*- coding: utf-8 -*-
"""
Tests for settings utilities.
"""
from unittest.mock import patch

import pytest
from nessai.flowsampler import FlowSampler
from nessai.utils.settings import (
    get_all_kwargs,
    get_run_kwargs_list,
    _get_importance_methods,
    _get_standard_methods,
)


def test_get_standard_methods():
    """Assert a list of methods is returned"""
    kwds, run = _get_standard_methods()
    assert len(kwds) == 4
    assert len(run) == 1


def test_get_importance_methods():
    """Assert a list of methods is returned"""
    kwds, run = _get_importance_methods()
    assert len(kwds) == 3
    assert len(run) == 1


@pytest.mark.parametrize(
    "ins, get_method",
    [(False, "_get_standard_methods"), (True, "_get_importance_methods")],
)
def test_get_all_kwargs(ins, get_method):
    """Assert the correct dictionary is returned.

    Positional arguments should be ignored.
    """

    def func0(a, b, c=2, d=None):
        pass

    def func1(e, f, g=3, h=True):
        pass

    expected = dict(c=2, d=None, g=3, h=True)

    with patch(
        f"nessai.utils.settings.{get_method}",
        return_value=[
            [
                func0,
            ],
            [
                func1,
            ],
        ],
    ):
        out = get_all_kwargs(importance_nested_sampler=ins, split_kwargs=False)

    assert out == expected


@pytest.mark.parametrize(
    "ins, get_method",
    [(False, "_get_standard_methods"), (True, "_get_importance_methods")],
)
def test_get_all_kwargs_split(ins, get_method):
    """Assert the correct dictionary is returned.

    Positional arguments should be ignored.
    """

    def func0(a, b, c=2, d=None):
        pass

    def func1(e, f, g=3, h=True):
        pass

    expected = (dict(c=2, d=None), dict(g=3, h=True))

    with patch(
        f"nessai.utils.settings.{get_method}",
        return_value=[
            [
                func0,
            ],
            [
                func1,
            ],
        ],
    ):
        out = get_all_kwargs(importance_nested_sampler=ins, split_kwargs=True)

    assert out == expected


@pytest.mark.parametrize(
    "ins, run_method",
    [(False, "run_standard_sampler"), (True, "run_importance_nested_sampler")],
)
def test_get_run_kwargs_list(ins, run_method):
    """Assert the correct list is returned"""

    def func(a, b=1, c=None):
        pass

    expected = ["b", "c"]

    with patch.object(FlowSampler, run_method, func):
        out = get_run_kwargs_list(importance_nested_sampler=ins)

    assert out == expected
