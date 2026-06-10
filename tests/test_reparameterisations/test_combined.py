# -*- coding: utf-8 -*-
"""
Test the combined reparameterisation class.
"""

from unittest.mock import MagicMock, call, create_autospec, patch

import numpy as np
import pytest

from nessai.livepoint import empty_structured_array
from nessai.reparameterisations import (
    Angle,
    CombinedReparameterisation,
    Reparameterisation,
    RescaleToBounds,
)
from nessai.utils.testing import assert_structured_arrays_equal


class PrimeProducer(Reparameterisation):
    def __init__(self):
        super().__init__(parameters="x", prior_bounds=None)
        self.prime_parameters = ["u"]

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        x_prime["u"] = 2.0 * x["x"]
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        x["x"] = x_prime["u"] / 2.0
        return x, x_prime, log_j


class PrimeConsumer(Reparameterisation):
    def __init__(self):
        super().__init__(parameters="y", prior_bounds=None)
        self.prime_parameters = ["v"]
        self.prime_requires = ["u"]

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        x_prime["v"] = x["y"] + x_prime["u"]
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        x["y"] = x_prime["v"] - x_prime["u"]
        return x, x_prime, log_j


@pytest.fixture
def reparam():
    # Use spec_set to raised an error if an unknown attribute is set
    # Use an instance so attributes from __init__ are included.
    return create_autospec(
        CombinedReparameterisation(), spec_set=True, instance=True
    )


def test_init_none():
    c = CombinedReparameterisation()
    assert c.parameters == []
    assert c.initial_parameters == []


def test_init_w_reparam():
    c = CombinedReparameterisation(
        RescaleToBounds(parameters="x", prior_bounds=[0, 1]),
        initial_parameters=["x"],
    )
    assert c.parameters == ["x"]
    assert c.prime_parameters == ["x_prime"]


@pytest.mark.parametrize("reverse_order", [False, True])
def test_to_prime_order(reverse_order, reparam):
    """Assert order is correct depending on the value of the reversed flag"""
    order = [1, 2, 3]
    reparam.order = order
    reparam.reverse_order = reverse_order

    expected = [1, 2, 3]
    if reverse_order:
        expected = list(reversed(expected))

    out = CombinedReparameterisation.to_prime_order.__get__(reparam)
    assert list(out) == expected


@pytest.mark.parametrize("reverse_order", [False, True])
def test_from_prime_order(reverse_order, reparam):
    """Assert order is correct depending on the value of the reversed flag"""
    order = [1, 2, 3]
    reparam.order = order
    reparam.reverse_order = reverse_order

    expected = [3, 2, 1]
    if reverse_order:
        expected = list(reversed(expected))

    out = CombinedReparameterisation.from_prime_order.__get__(reparam)
    assert list(out) == expected


def test_add_single_reparameterisations():
    """Test the core functionality of adding reparameterisations"""
    r = RescaleToBounds(parameters="x", prior_bounds=[0, 1])
    c = CombinedReparameterisation(initial_parameters=["x"])
    c.add_reparameterisations(r)
    assert c.parameters == ["x"]


def test_add_single_reparameterisations_with_auxiliary_parameter(LightReparam):
    r = LightReparam(
        "a", parameters=["x"], requires=[], auxiliary_parameters=["aux"]
    )
    c = CombinedReparameterisation(initial_parameters=["x"])
    c.add_reparameterisations(r)
    assert c.parameters == ["x", "aux"]


def test_add_multiple_reparameterisations(model):
    """
    Test adding multiple reparameterisations and using the reparameterisation.
    """
    r = [
        RescaleToBounds(parameters="x", prior_bounds=model.bounds["x"]),
        RescaleToBounds(parameters="y", prior_bounds=model.bounds["y"]),
    ]
    reparam = CombinedReparameterisation(initial_parameters=["x", "y"])
    reparam.add_reparameterisations(r)

    assert reparam.parameters == ["x", "y"]

    n = 100
    x = model.new_point(N=n)
    x_prime = empty_structured_array(n, names=reparam.prime_parameters)
    log_j = 0

    x_re, x_prime_re, log_j_re = reparam.reparameterise(x, x_prime, log_j)

    assert_structured_arrays_equal(x, x_re)

    x_in = empty_structured_array(n, names=reparam.parameters)

    x_inv, x_prime_inv, log_j_inv = reparam.inverse_reparameterise(
        x_in, x_prime_re, log_j
    )

    assert_structured_arrays_equal(x, x_inv, atol=1e-15, rtol=1e-15)
    assert_structured_arrays_equal(
        x_prime_re, x_prime_inv, atol=1e-15, rtol=1e-15
    )
    np.testing.assert_array_equal(log_j_re, -log_j_inv)


def test_base_add_reparameterisation_missing_requirements(reparam):
    """Assert an error is raised if a required parameter is missing"""
    r = MagicMock(spec=Reparameterisation)
    r.parameters = ["y"]
    r.requires = ["x"]
    r.input_parameters = ["y", "x"]
    r.output_parameters = ["y"]
    r.prime_requires = []

    reparam.parameters = ["y"]
    reparam.prime_parameters = ["y_prime"]
    reparam.initial_parameters = []

    with pytest.raises(RuntimeError) as excinfo:
        CombinedReparameterisation._add_reparameterisation(reparam, r)
    assert "missing requirement(s)" in str(excinfo.value)


def test_add_reparameterisation(reparam):
    """Assert add_reparameterisation calls the correct method.

    Should just call CombinedReparameterisation.add_reparameteristions
    """
    new_reparam = MagicMock(spec=Reparameterisation)
    CombinedReparameterisation.add_reparameterisation(reparam, new_reparam)
    reparam.add_reparameterisations.assert_called_once_with(new_reparam)


def test_add_reparameterisations_single(reparam):
    """Assert add_reparameterisations works correctly for a single reparam.

    Should convert to a list and then pass to `_add_reparameterisation`
    """
    reparam.parameters = []
    reparam.initial_parameters = []
    reparam.prime_parameters = []
    new_reparam = MagicMock(spec=Reparameterisation)
    with patch(
        "nessai.reparameterisations.combined.sort_reparameterisations",
        return_value=[new_reparam],
    ) as mock:
        CombinedReparameterisation.add_reparameterisations(
            reparam, new_reparam
        )
    reparam._add_reparameterisation.assert_called_once_with(new_reparam)
    mock.assert_called_once_with(
        [new_reparam], existing_parameters=[], existing_prime_parameters=[]
    )


def test_add_reparameterisations_multiple(reparam):
    """Assert add_reparameterisations works correctly with multiple reparams.

    Should call `_add_reparmeterisation` with each reparam.
    """
    parameters = ["x"]
    reparam.parameters = parameters
    reparam.initial_parameters = []
    reparam.prime_parameters = []
    r1 = MagicMock(spec=Reparameterisation)
    r2 = MagicMock(spec=Reparameterisation)
    reparams = [r1, r2]
    with patch(
        "nessai.reparameterisations.combined.sort_reparameterisations",
        return_value=[r1, r2],
    ) as mock:
        CombinedReparameterisation.add_reparameterisations(reparam, reparams)
    reparam._add_reparameterisation.assert_has_calls([call(r1), call(r2)])
    mock.assert_called_once_with(
        reparams,
        existing_parameters=parameters,
        existing_prime_parameters=[],
    )


def test_add_reparameterisations_with_prime_requirement(LightReparam):
    r0 = LightReparam(
        "a", parameters=["x"], requires=[], prime_parameters=["u"]
    )
    r1 = LightReparam(
        "b",
        parameters=["y"],
        requires=[],
        prime_parameters=["v"],
        prime_requires=["u"],
    )

    c = CombinedReparameterisation(initial_parameters=["x", "y"])
    c.add_reparameterisations([r1, r0])

    assert c.order == ["a", "b"]


def test_prime_requires_chain():
    c = CombinedReparameterisation(initial_parameters=["x", "y"])
    c.add_reparameterisations([PrimeConsumer(), PrimeProducer()])

    assert c.order == ["primeproducer_x", "primeconsumer_y"]
    assert c.prime_parameters == ["u", "v"]

    x = empty_structured_array(4, names=["x", "y"])
    x["x"] = np.array([0.5, 1.0, 1.5, 2.0])
    x["y"] = np.array([1.0, 2.0, 3.0, 4.0])
    x_prime = empty_structured_array(4, names=c.prime_parameters)
    log_j = np.zeros(x.size)

    x_re, x_prime_re, log_j_re = c.reparameterise(x, x_prime, log_j)
    assert_structured_arrays_equal(x_re, x)
    np.testing.assert_allclose(x_prime_re["u"], 2.0 * x["x"])
    np.testing.assert_allclose(x_prime_re["v"], x["y"] + x_prime_re["u"])

    x_in = empty_structured_array(4, names=c.parameters)
    x_inv, x_prime_inv, log_j_inv = c.inverse_reparameterise(
        x_in, x_prime_re, log_j
    )

    assert_structured_arrays_equal(x_inv, x)
    assert_structured_arrays_equal(x_prime_inv, x_prime_re)
    np.testing.assert_array_equal(log_j_re, log_j_inv)


def test_base_add_reparameterisation_missing_prime_requirements(reparam):
    """Assert an error is raised if a prime requirement is missing"""
    r = MagicMock(spec=Reparameterisation)
    r.parameters = ["y"]
    r.requires = []
    r.input_parameters = ["y"]
    r.output_parameters = ["y"]
    r.prime_requires = ["x_prime"]

    reparam.parameters = ["y"]
    reparam.prime_parameters = ["y_prime"]
    reparam.initial_parameters = ["y"]

    with pytest.raises(RuntimeError) as excinfo:
        CombinedReparameterisation._add_reparameterisation(reparam, r)
    assert "Missing prime requirement(s)" in str(excinfo.value)


def test_check_order_valid(reparam, LightReparam):
    """Assert no error is raised if the order is valid"""
    d = dict(
        a=LightReparam(
            "a", parameters=["x"], requires=[], inverse_requires=[]
        ),
        b=LightReparam(
            "b", parameters=["y"], requires=["x"], inverse_requires=["x"]
        ),
        c=LightReparam(
            "c", parameters=["z"], requires=["y"], inverse_requires=["y"]
        ),
    )
    reparam.__getitem__.side_effect = d.__getitem__
    reparam.from_prime_order = ["a", "b", "c"]

    CombinedReparameterisation.check_order(reparam)


def test_check_order_invalid(reparam, LightReparam):
    """Assert an error is raised if the order is invalid"""
    d = dict(
        a=LightReparam(
            "a", parameters=["x"], requires=["z"], inverse_requires=["w"]
        ),
        b=LightReparam(
            "b", parameters=["y"], requires=[], inverse_requires=[]
        ),
        c=LightReparam(
            "c", parameters=["z"], requires=["y"], inverse_requires=[]
        ),
    )
    reparam.__getitem__.side_effect = d.__getitem__
    reparam.from_prime_order = ["a", "b", "c"]
    print(reparam["a"])

    with pytest.raises(
        RuntimeError, match="Order of reparameterisations is invalid *."
    ):
        CombinedReparameterisation.check_order(reparam)


def test_update_bounds(reparam):
    """Assert update bounds calls the method for each reparam"""
    x = [1, 2]
    # Angle doesn't have update bounds
    r1 = MagicMock(spec=Angle)
    # RescaleToBounds does have update bounds
    r2 = MagicMock(spec=RescaleToBounds)
    reparam.values = MagicMock(return_value=[r1, r2])

    assert not hasattr(r1, "update_bounds")

    CombinedReparameterisation.update_bounds(reparam, x)

    r2.update_bounds.assert_called_once_with(x)


def test_reset_inversion(reparam):
    """Assert reset inversion calls the method for each reparam"""
    # Angle doesn't have reset inversion
    r1 = MagicMock(spec=Angle)
    # RescaleToBounds does have reset inversion
    r2 = MagicMock(spec=RescaleToBounds)
    reparam.values = MagicMock(return_value=[r1, r2])

    assert not hasattr(r1, "reset_inversion")

    CombinedReparameterisation.reset_inversion(reparam)

    r2.reset_inversion.assert_called_once()


def test_update(reparam):
    """Assert update calls the update method for all reparameterisations."""
    r1 = MagicMock(spec=Angle)
    r2 = MagicMock(spec=RescaleToBounds)
    reparam.values = MagicMock(return_value=[r1, r2])

    x = np.array((1, 2), dtype=[("x", "f8"), ("y", "f8")])

    CombinedReparameterisation.update(reparam, x)

    r1.update.assert_called_once_with(x)
    r2.update.assert_called_once_with(x)


def test_log_prior(reparam):
    """Assert log_prior is only called for reparams with has_prior==True"""
    r1 = MagicMock(spec=Reparameterisation)
    r2 = MagicMock(spec=Reparameterisation)
    r3 = MagicMock(spec=Reparameterisation)
    r1.has_prior = False
    r2.has_prior = True
    r3.has_prior = True
    r1.log_prior = MagicMock()
    r2.log_prior = MagicMock(return_value=-5)
    r3.log_prior = MagicMock(return_value=-6)
    reparam.values = MagicMock(return_value=[r1, r2, r3])

    x = np.array([(1, 2)], dtype=[("x", "f8"), ("y", "f8")])

    out = CombinedReparameterisation.log_prior(reparam, x)

    assert out == -11
    r1.log_prior.assert_not_called()
    r2.log_prior.assert_called_once_with(x)
    r3.log_prior.assert_called_once_with(x)
