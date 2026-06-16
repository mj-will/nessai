"""Configuration for the reparameterisation tests"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest

from nessai.livepoint import empty_structured_array
from nessai.utils.testing import assert_structured_arrays_equal


@dataclass(init=False)
class _LightReparam:
    """Reparameterisation-like object"""

    name: str
    input_parameters: List[str]
    output_parameters: List[str] = field(default_factory=list)
    persistent_parameters: List[str] = field(default_factory=list)
    auxiliary_parameters: List[str] = field(default_factory=list)
    inverse_input_parameters: List[str] = field(default_factory=list)

    def __init__(
        self,
        name,
        input_parameters=None,
        output_parameters=None,
        persistent_parameters=None,
        auxiliary_parameters=None,
        inverse_input_parameters=None,
        parameters=None,
    ) -> None:
        if input_parameters is None:
            input_parameters = parameters
        if input_parameters is None:
            input_parameters = []

        self.name = name
        self.input_parameters = list(input_parameters)
        self.output_parameters = list(output_parameters or [])
        self.persistent_parameters = list(persistent_parameters or [])
        self.auxiliary_parameters = list(auxiliary_parameters or [])
        self.inverse_input_parameters = list(inverse_input_parameters or [])
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.output_parameters:
            self.output_parameters = [
                p + "_prime" for p in self.input_parameters
            ]
        self._x_input_parameters = []
        self._x_prime_input_parameters = []
        self._x_persistent_parameters = []
        self._x_prime_persistent_parameters = []
        self._x_inverse_input_parameters = []
        self._x_prime_inverse_input_parameters = []

    @property
    def parameters(self) -> List[str]:
        return self.input_parameters

    @parameters.setter
    def parameters(self, value) -> None:
        self.input_parameters = value

    @property
    def x_input_parameters(self) -> List[str]:
        return self._x_input_parameters or self.input_parameters

    @property
    def x_prime_input_parameters(self) -> List[str]:
        return self._x_prime_input_parameters

    @property
    def x_persistent_parameters(self) -> List[str]:
        return self._x_persistent_parameters

    @property
    def x_prime_persistent_parameters(self) -> List[str]:
        return self._x_prime_persistent_parameters

    @property
    def x_output_parameters(self) -> List[str]:
        return self.input_parameters + self.auxiliary_parameters

    def resolve_forward_input_spaces(self, parameters, prime_parameters):
        self._x_input_parameters = []
        self._x_prime_input_parameters = []
        missing = []
        for p in self.input_parameters:
            if p in parameters:
                self._x_input_parameters.append(p)
            elif p in prime_parameters:
                self._x_prime_input_parameters.append(p)
            else:
                missing.append(p)
        self._x_persistent_parameters = [
            p
            for p in self.persistent_parameters
            if p in self._x_input_parameters
        ]
        self._x_prime_persistent_parameters = [
            p
            for p in self.persistent_parameters
            if p in self._x_prime_input_parameters
        ]
        return missing

    def resolve_inverse_input_spaces(self, parameters, prime_parameters):
        self._x_inverse_input_parameters = []
        self._x_prime_inverse_input_parameters = []
        missing = []
        for p in self.inverse_input_parameters:
            if p in parameters:
                self._x_inverse_input_parameters.append(p)
            elif p in prime_parameters:
                self._x_prime_inverse_input_parameters.append(p)
            else:
                missing.append(p)
        return missing


@pytest.fixture
def LightReparam() -> _LightReparam:
    return _LightReparam


@pytest.fixture()
def is_invertible(model, n=100):
    """Test if a reparameterisation is invertible."""

    def test_invertibility(
        reparam, model=model, atol=0.0, rtol=0.0, unit_hypercube=False
    ):
        if unit_hypercube:
            x = model.sample_unit_hypercube(n)
        else:
            x = model.new_point(N=n)
        x_prime = empty_structured_array(n, names=reparam.output_parameters)
        log_j = np.zeros(n)
        assert x.size == x_prime.size

        x_re, x_prime_re, log_j_re = reparam.reparameterise(x, x_prime, log_j)

        x_in = empty_structured_array(x_re.size, reparam.x_output_parameters)
        log_j = np.zeros(x_re.size)

        x_inv, x_prime_inv, log_j_inv = reparam.inverse_reparameterise(
            x_in, x_prime_re, log_j
        )

        assert_structured_arrays_equal(x_inv, x, atol=atol, rtol=rtol)
        np.testing.assert_allclose(-log_j_inv, log_j_re)

        return True

    return test_invertibility
