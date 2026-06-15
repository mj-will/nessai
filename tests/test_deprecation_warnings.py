# -*- coding: utf-8 -*-
"""
Tests for modules/functions that are soon to be deprecated.
"""

from unittest.mock import create_autospec

import pytest


def test_compute_evidence_ratio_deprecation():
    """Assert a warning is raised when compute_evidence_ratio is called"""
    from nessai.evidence import _INSIntegralState

    state = create_autospec(_INSIntegralState)
    with pytest.deprecated_call():
        _INSIntegralState.compute_evidence_ratio(state)


def test_rescaled_dims_deprecation():
    """Assert a warning is raised when rescaled_dims is accessed"""
    from nessai.proposal.flowproposal import FlowProposal

    proposal = create_autospec(FlowProposal, prime_parameters=["x", "y"])
    with pytest.deprecated_call():
        assert FlowProposal.rescaled_dims.__get__(proposal) == 2
