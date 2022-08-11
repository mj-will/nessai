# -*- coding: utf-8 -*-
"""
General tests of `get_reparameterisation`.
"""
import pytest

from nessai.reparameterisations import (
    Angle,
    AnglePair,
    NullReparameterisation,
    Reparameterisation,
    Rescale,
    RescaleToBounds,
    ToCartesian,
    get_reparameterisation,
)

# List of known reparameterisations, the class and expected kwargs.
known_reparameteristions = [
    ("default", RescaleToBounds, {}),
    ("rescaletobounds", RescaleToBounds, {}),
    ("rescale-to-bounds", RescaleToBounds, {}),
    ("offset", RescaleToBounds, {"offset": True}),
    (
        "inversion",
        RescaleToBounds,
        {
            "detect_edges": True,
            "boundary_inversion": True,
            "inversion_type": "split",
        },
    ),
    (
        "inversion-duplicate",
        RescaleToBounds,
        {
            "detect_edges": True,
            "boundary_inversion": True,
            "inversion_type": "duplicate",
        },
    ),
    (
        "logit",
        RescaleToBounds,
        {
            "update_bounds": False,
            "rescale_bounds": [0.0, 1.0],
            "post_rescaling": "logit",
        },
    ),
    ("scale", Rescale, {}),
    ("rescale", Rescale, {}),
    ("angle", Angle, {}),
    ("angle-pi", Angle, {"scale": 2.0, "prior": "uniform"}),
    ("angle-2pi", Angle, {"scale": 1.0, "prior": "uniform"}),
    ("periodic", Angle, {"scale": None}),
    ("angle-sine", RescaleToBounds, {}),
    ("angle-cosine", RescaleToBounds, {}),
    ("angle-pair", AnglePair, {}),
    ("to-cartesian", ToCartesian, {}),
    ("none", NullReparameterisation, {}),
    ("null", NullReparameterisation, {}),
    (None, NullReparameterisation, {}),
]


@pytest.mark.parametrize(
    "inputs",
    known_reparameteristions,
)
def test_get_reparameterisation(inputs):
    """Test all of the included reparameterisations."""
    name, expected_class, expected_kwargs = inputs
    reparam, kwargs = get_reparameterisation(name)
    assert reparam is expected_class
    assert kwargs == expected_kwargs


def test_get_reparameterisation_with_class():
    """Test the case of class that inherits from Reparameterisation"""

    class TestReparam(Reparameterisation):
        pass

    reparam, kwargs = get_reparameterisation(TestReparam)
    assert reparam is TestReparam
    assert not kwargs


def test_get_reparameterisation_with_class_error():
    """Test the case of class that doe not inherit from Reparameterisation.

    This should raise an error.
    """

    class TestReparam:
        pass

    with pytest.raises(TypeError) as excinfo:
        get_reparameterisation(TestReparam)
    assert "Reparameterisation must be" in str(excinfo.value)


def test_get_reparameterisation_unknown_name():
    """Assert an error is raised in the string is not recognised."""
    with pytest.raises(ValueError) as excinfo:
        get_reparameterisation("shift")
    assert "Unknown reparameterisation: shift" in str(excinfo.value)


def test_get_reparameterisation_invalid_input():
    """Assert an error is raised if the input type is invalid."""
    with pytest.raises(TypeError) as excinfo:
        get_reparameterisation(2)
    assert "Reparameterisation must be" in str(excinfo.value)


def test_get_reparameterisation_defaults():
    """Test using an updated defaults dictionary."""
    defaults = {"default": ("Class", {"x": 2})}
    cls, kwargs = get_reparameterisation("default", defaults=defaults)
    assert cls == "Class"
    assert kwargs == {"x": 2}
