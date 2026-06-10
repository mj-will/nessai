"""Configuration for the reparameterisation tests"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest

from nessai.livepoint import empty_structured_array
from nessai.utils.testing import assert_structured_arrays_equal


@dataclass
class _LightReparam:
    """Reparameterisation-like object"""

    name: str
    parameters: List[str]
    requires: List[str]
    auxiliary_parameters: List[str] = field(default_factory=list)
    prime_parameters: List[str] = field(default_factory=list)
    prime_requires: List[str] = field(default_factory=list)
    inverse_requires: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.prime_parameters:
            self.prime_parameters = [p + "_prime" for p in self.parameters]

    @property
    def output_parameters(self) -> List[str]:
        return self.parameters + self.auxiliary_parameters

    @property
    def input_parameters(self) -> List[str]:
        return list(dict.fromkeys(self.parameters + self.requires))


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
        x_prime = empty_structured_array(n, names=reparam.prime_parameters)
        log_j = np.zeros(n)
        assert x.size == x_prime.size

        x_re, x_prime_re, log_j_re = reparam.reparameterise(x, x_prime, log_j)

        x_in = empty_structured_array(x_re.size, reparam.parameters)
        log_j = np.zeros(x_re.size)

        x_inv, x_prime_inv, log_j_inv = reparam.inverse_reparameterise(
            x_in, x_prime_re, log_j
        )

        assert_structured_arrays_equal(x_inv, x, atol=atol, rtol=rtol)
        np.testing.assert_allclose(-log_j_inv, log_j_re)

        return True

    return test_invertibility
