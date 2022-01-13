# -*- coding: utf-8 -*-
"""Tests for GW reparameterisations"""

import pytest
from nessai.gw.reparameterisations import (
    DistanceReparameterisation,
    default_gw,
    get_gw_reparameterisation,
)
from unittest.mock import patch


def test_get_gw_reparameterisation():
    """Test getting the gw reparameterisations.

    Assert the correct defaults are used.
    """
    expected = 'out'
    with patch(
        'nessai.gw.reparameterisations.get_reparameterisation',
        return_value=expected,
    ) as base_fn:
        out = get_gw_reparameterisation('mass_ratio')
    assert out == expected
    base_fn.assert_called_once_with('mass_ratio', defaults=default_gw)


@pytest.mark.integration_test
def test_get_gw_reparameterisation_integration():
    """Integration test for get_gw_reparameterisation"""
    reparam, _ = get_gw_reparameterisation('distance')
    assert reparam is DistanceReparameterisation
