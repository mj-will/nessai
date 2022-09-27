# -*- coding: utf-8 -*-
"""
Tests for modules/functions that are soon to be deprecated.
"""
from nessai.utils import configure_threads
import pytest


def test_max_threads_warning():
    """Assert a future warning is raised if max threads is specified"""
    with pytest.warns(FutureWarning) as record:
        configure_threads(max_threads=1)
    assert "`max_threads` is deprecated" in str(record[0].message)


def test_nested_sampler_deprecation():
    """Assert a warning is raised with nessai.nestedsampler is imported."""
    with pytest.warns(FutureWarning) as record:
        from nessai import nestedsampler  # noqa
    assert "`nessai.nestedsampler` is deprecated" in str(record[0].message)


def test_lulinear_warning():
    """Assert a warning is raised when LULinear is imported"""
    with pytest.warns(FutureWarning) as record:
        from nessai.flows.transforms import LULinear  # noqa
    assert "`nessai.flows.transforms.LULinear` is deprecated" in str(
        record[0].message
    )
