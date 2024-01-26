# -*- coding: utf-8 -*-
"""
Tests for modules/functions that are soon to be deprecated.
"""
import pytest


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


def test_bilbyutils_warning():
    """Assert a warning is raised if bilbyutils is imported"""
    with pytest.warns(
        FutureWarning, match=r"`nessai.utils.bilbyutils` is deprecated"
    ):
        from nessai.utils.bilbyutils import get_all_kwargs  # noqa
