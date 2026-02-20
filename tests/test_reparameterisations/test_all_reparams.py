from inspect import signature

import pytest

from nessai.reparameterisations import (
    default_reparameterisations,
)


@pytest.fixture(params=default_reparameterisations.values())
def reparameterisation(request):
    """Fixture to test all reparameterisations."""
    return request.param


def test_reparams_rng(reparameterisation):
    """Test that all reparameterisations can be initialised with an rng"""
    sig = signature(reparameterisation.class_fn)
    assert "rng" in sig.parameters
