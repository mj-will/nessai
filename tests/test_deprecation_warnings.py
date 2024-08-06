# -*- coding: utf-8 -*-
"""
Tests for modules/functions that are soon to be deprecated.
"""
import pytest
from unittest.mock import create_autospec


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


def test_flowproposal_names_warning():
    from nessai.proposal import FlowProposal

    proposal = create_autospec(FlowProposal)
    proposal.parameters = ["x"]
    with pytest.warns(FutureWarning, match=r"`names` is deprecated"):
        assert FlowProposal.names.__get__(proposal) == ["x"]


def test_flowproposal_rescaled_names_warning():
    from nessai.proposal import FlowProposal

    proposal = create_autospec(FlowProposal)
    proposal.prime_parameters = ["x"]
    with pytest.warns(FutureWarning, match=r"`rescaled_names` is deprecated"):
        assert FlowProposal.rescaled_names.__get__(proposal) == ["x"]


def test_flowproposal_update_bounds_warning():
    from nessai.proposal import FlowProposal

    proposal = create_autospec(FlowProposal)
    proposal.should_update_reparameterisations = True
    with pytest.warns(FutureWarning, match=r"`update_bounds` is deprecated"):
        assert FlowProposal.update_bounds.__get__(proposal) is True
