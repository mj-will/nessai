"""Tests for the sorting utilities"""

from dataclasses import dataclass, field
from typing import List

import pytest

from nessai.utils.sorting import sort_reparameterisations


@dataclass
class Reparameterisation:
    """A dataclass that mocks the Reparameterisation class"""

    name: str
    input_parameters: List[str]
    output_parameters: List[str] = field(default_factory=list)
    persistent_parameters: List[str] = field(default_factory=list)
    auxiliary_parameters: List[str] = field(default_factory=list)
    inverse_input_parameters: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.output_parameters:
            self.output_parameters = [
                p + "_prime" for p in self.input_parameters
            ]
        self._x_input_parameters = []
        self._x_prime_input_parameters = []
        self._x_persistent_parameters = []
        self._x_prime_persistent_parameters = []

    @property
    def parameters(self):
        return self.input_parameters

    @property
    def prime_parameters(self):
        return self.output_parameters

    @property
    def x_input_parameters(self):
        return self._x_input_parameters or self.input_parameters

    @property
    def x_prime_input_parameters(self):
        return self._x_prime_input_parameters

    @property
    def x_output_parameters(self):
        return self.x_input_parameters + self.auxiliary_parameters

    def resolve_forward_input_spaces(self, parameters, prime_parameters):
        self._x_input_parameters = []
        self._x_prime_input_parameters = []
        self._x_persistent_parameters = []
        self._x_prime_persistent_parameters = []
        missing = []
        for p in self.input_parameters:
            if p in parameters:
                self._x_input_parameters.append(p)
            elif p in prime_parameters:
                self._x_prime_input_parameters.append(p)
            else:
                missing.append(p)
        return missing


def test_sorting():
    """Test a basic sorting example"""
    r0 = Reparameterisation("1", ["a", "b", "c"], ["a_p"])
    r1 = Reparameterisation("2", ["b", "c"], ["b_p"])
    r2 = Reparameterisation("3", ["c"], ["c_p"])

    out = sort_reparameterisations(
        [r0, r1, r2], existing_parameters=["a", "b", "c"]
    )
    print([o.name for o in out])
    assert out == [r2, r1, r0]


def test_sorting_existing():
    """Test sorting when there are existing parameters"""
    r0 = Reparameterisation("1", ["a", "c"], ["a_p"])
    r1 = Reparameterisation("2", ["b"], ["b_p"])
    r2 = Reparameterisation("3", ["c", "d"], ["c_p"])

    out = sort_reparameterisations(
        [r0, r1, r2], existing_parameters=["a", "b", "c", "d"]
    )
    print([o.name for o in out])
    assert out == [r1, r0, r2]


def test_sorting_error():
    """Assert an errors is raised if there is an invalid requirement"""
    r0 = Reparameterisation("1", ["a"], ["a_p"])
    r1 = Reparameterisation("2", ["b", "c"], ["b_p"])
    with pytest.raises(ValueError, match=r"requires inputs \['c'\]"):
        sort_reparameterisations([r0, r1], existing_parameters=["a", "b"])


def test_sorting_with_auxiliary_parameters():
    r0 = Reparameterisation("1", ["a"], ["a_p"], auxiliary_parameters=["aux"])
    r1 = Reparameterisation("2", ["b", "aux"], ["b_p"])

    out = sort_reparameterisations([r1, r0], existing_parameters=["a", "b"])

    assert out == [r0, r1]


def test_sorting_with_prime_requirements():
    r0 = Reparameterisation("1", ["a"], ["a_p"])
    r1 = Reparameterisation("2", ["b", "a_p"], ["b_p"])

    out = sort_reparameterisations([r1, r0], existing_parameters=["a", "b"])

    assert out == [r0, r1]


def test_sorting_with_unknown_prime_requirement():
    r0 = Reparameterisation("1", ["a"], ["a_p"])
    r1 = Reparameterisation("2", ["b", "c_p"], ["b_p"])

    with pytest.raises(ValueError, match=r"requires inputs \['c_p'\]"):
        sort_reparameterisations([r0, r1], existing_parameters=["a", "b"])


def test_sorting_retries_skipped_reparameterisations():
    """Assert skipped reparameterisations are retried after one pass.

    The two reparameterisations have the same initial sort weight, so the
    first pass keeps the input order. `r0` is skipped because it depends on
    an auxiliary parameter that is only added once `r1` has been applied.
    """
    r0 = Reparameterisation("1", ["a", "aux"], ["a_p"])
    r1 = Reparameterisation(
        "2",
        ["b", "seed"],
        ["b_p"],
        auxiliary_parameters=["aux"],
    )

    out = sort_reparameterisations(
        [r0, r1], existing_parameters=["a", "b", "seed"]
    )

    assert out == [r1, r0]


def test_sorting_error_if_skipped_and_no_progress():
    r0 = Reparameterisation(
        "1", ["a", "aux_b"], ["a_p"], auxiliary_parameters=["aux_a"]
    )
    r1 = Reparameterisation(
        "2", ["b", "aux_a"], ["b_p"], auxiliary_parameters=["aux_b"]
    )

    with pytest.raises(ValueError, match="Could not sort reparameterisations"):
        sort_reparameterisations([r0, r1], existing_parameters=["a", "b"])
