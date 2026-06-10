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
