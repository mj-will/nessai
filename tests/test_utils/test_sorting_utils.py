"""Tests for the sorting utilities"""

from dataclasses import dataclass, field
from typing import List

import pytest

from nessai.utils.sorting import sort_reparameterisations


@dataclass
class Reparameterisation:
    """A dataclass that mocks the Reparameterisation class"""

    name: str
    parameters: List[str]
    requires: List[str]
    auxiliary_parameters: List[str] = field(default_factory=list)
    prime_parameters: List[str] = field(default_factory=list)
    prime_requires: List[str] = field(default_factory=list)
    inverse_requires: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.prime_parameters:
            self.prime_parameters = [p + "_prime" for p in self.parameters]

    @property
    def output_parameters(self):
        return self.parameters + self.auxiliary_parameters

    @property
    def input_parameters(self):
        return list(dict.fromkeys(self.parameters + self.requires))


def test_sorting():
    """Test a basic sorting example"""
    r0 = Reparameterisation("1", ["a"], requires=["b", "c"])
    r1 = Reparameterisation("2", ["b"], requires=["c"])
    r2 = Reparameterisation("3", ["c"], requires=[])

    out = sort_reparameterisations(
        [r0, r1, r2], existing_parameters=["a", "b", "c"]
    )
    print([o.name for o in out])
    assert out == [r2, r1, r0]


def test_sorting_existing():
    """Test sorting when there are existing parameters"""
    r0 = Reparameterisation("1", ["a"], requires=["c"])
    r1 = Reparameterisation("2", ["b"], requires=[])
    r2 = Reparameterisation("3", ["c"], requires=["d"])

    out = sort_reparameterisations(
        [r0, r1, r2], existing_parameters=["a", "b", "c", "d"]
    )
    print([o.name for o in out])
    assert out == [r1, r0, r2]


def test_sorting_error():
    """Assert an errors is raised if there is an invalid requirement"""
    r0 = Reparameterisation("1", ["a"], requires=[])
    r1 = Reparameterisation("2", ["b"], requires=["c"])
    with pytest.raises(ValueError, match=r".* are not known \(\['a', 'b'\]\)"):
        sort_reparameterisations([r0, r1], existing_parameters=["a", "b"])


def test_sorting_with_auxiliary_parameters():
    r0 = Reparameterisation(
        "1", ["a"], requires=[], auxiliary_parameters=["aux"]
    )
    r1 = Reparameterisation("2", ["b"], requires=["aux"])

    out = sort_reparameterisations([r1, r0], existing_parameters=["a", "b"])

    assert out == [r0, r1]


def test_sorting_with_prime_requirements():
    r0 = Reparameterisation("1", ["a"], requires=[], prime_parameters=["a_p"])
    r1 = Reparameterisation(
        "2",
        ["b"],
        requires=[],
        prime_parameters=["b_p"],
        prime_requires=["a_p"],
    )

    out = sort_reparameterisations([r1, r0], existing_parameters=["a", "b"])

    assert out == [r0, r1]


def test_sorting_with_unknown_prime_requirement():
    r0 = Reparameterisation("1", ["a"], requires=[], prime_parameters=["a_p"])
    r1 = Reparameterisation(
        "2",
        ["b"],
        requires=[],
        prime_parameters=["b_p"],
        prime_requires=["c_p"],
    )

    with pytest.raises(
        ValueError, match=r"requires prime parameters \['c_p'\]"
    ):
        sort_reparameterisations([r0, r1], existing_parameters=["a", "b"])


def test_sorting_retries_skipped_reparameterisations():
    """Assert skipped reparameterisations are retried after one pass.

    The two reparameterisations have the same initial sort weight, so the
    first pass keeps the input order. `r0` is skipped because it depends on
    an auxiliary parameter that is only added once `r1` has been applied.
    """
    r0 = Reparameterisation("1", ["a"], requires=["aux"])
    r1 = Reparameterisation(
        "2",
        ["b"],
        requires=["seed"],
        auxiliary_parameters=["aux"],
    )

    out = sort_reparameterisations(
        [r0, r1], existing_parameters=["a", "b", "seed"]
    )

    assert out == [r1, r0]


def test_sorting_error_if_skipped_and_no_progress():
    r0 = Reparameterisation("1", ["a"], requires=["b"])
    r1 = Reparameterisation("2", ["b"], requires=["a"])

    with pytest.raises(ValueError, match="Could not sort reparameterisations"):
        sort_reparameterisations([r0, r1])
