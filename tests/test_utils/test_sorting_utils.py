"""Tests for the sorting utilities"""
from dataclasses import dataclass
from typing import List

from nessai.utils.sorting import sort_reparameterisations
import pytest


@dataclass
class Reparameterisation:
    """A dataclass that mocks the Reparameterisation class"""

    name: str
    parameters: List[str]
    requires: List[str]


def test_sorting():
    """Test a basic sorting example"""
    r0 = Reparameterisation("1", ["a"], requires=["b", "c"])
    r1 = Reparameterisation("2", ["b"], requires=["c"])
    r2 = Reparameterisation("3", ["c"], requires=[])

    out = sort_reparameterisations([r0, r1, r2])
    print([o.name for o in out])
    assert out == [r2, r1, r0]


def test_sorting_existing():
    """Test sorting when there are existing parameters"""
    r0 = Reparameterisation("1", ["a"], requires=["c"])
    r1 = Reparameterisation("2", ["b"], requires=[])
    r2 = Reparameterisation("3", ["c"], requires=["d"])

    out = sort_reparameterisations([r0, r1, r2], existing_parameters=["d"])
    print([o.name for o in out])
    assert out == [r1, r2, r0]


def test_sorting_error():
    """Assert an errors is raised if there is an invalid requirement"""
    r0 = Reparameterisation("1", ["a"], requires=[])
    r1 = Reparameterisation("2", ["b"], requires=["c"])
    with pytest.raises(ValueError, match=r".* are not known \(\['a', 'b'\]\)"):
        sort_reparameterisations([r0, r1])
