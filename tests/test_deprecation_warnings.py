# -*- coding: utf-8 -*-
"""
Tests for modules/functions that are soon to be deprecated.
"""

from unittest.mock import create_autospec

import pytest


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


def test_compute_evidence_ratio_deprecation():
    """Assert a warning is raised when compute_evidence_ratio is called"""
    from nessai.evidence import _INSIntegralState

    state = create_autospec(_INSIntegralState)
    with pytest.deprecated_call():
        _INSIntegralState.compute_evidence_ratio(state)
