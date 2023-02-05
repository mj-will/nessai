# -*- coding: utf-8 -*-
"""
Test the combined reparameterisation class.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, call, create_autospec, patch

from nessai.reparameterisations import (
    Angle,
    CombinedReparameterisation,
    Reparameterisation,
    RescaleToBounds,
)
from nessai.livepoint import empty_structured_array
from nessai.utils.testing import assert_structured_arrays_equal


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


def test_init_w_reparam():
    c = CombinedReparameterisation(RescaleToBounds("x", [0, 1]))
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
    c = CombinedReparameterisation()
    c.add_reparameterisations(r)
    assert c.parameters == ["x"]
    assert c.has_prime_prior is False


def test_add_multiple_reparameterisations(model):
    """
    Test adding multiple reparameterisations and using the reparameterisation.
    """
    r = [
        RescaleToBounds(
            parameters="x", prior_bounds=model.bounds["x"], prior="uniform"
        ),
        RescaleToBounds(
            parameters="y", prior_bounds=model.bounds["y"], prior="uniform"
        ),
    ]
    reparam = CombinedReparameterisation()
    reparam.add_reparameterisations(r)

    assert reparam.parameters == ["x", "y"]
    assert reparam.has_prime_prior is True

    n = 100
    x = model.new_point(N=n)
    x_prime = empty_structured_array(n, names=reparam.prime_parameters)
    log_j = 0

    x_re, x_prime_re, log_j_re = reparam.reparameterise(x, x_prime, log_j)

    assert reparam.x_prime_log_prior(x_prime_re) is not None

    assert_structured_arrays_equal(x, x_re)

    x_in = empty_structured_array(n, names=reparam.parameters)

    x_inv, x_prime_inv, log_j_inv = reparam.inverse_reparameterise(
        x_in, x_prime_re, log_j
    )

    assert_structured_arrays_equal(x, x_inv)
    assert_structured_arrays_equal(x_prime_re, x_prime_inv)
    np.testing.assert_array_equal(log_j_re, -log_j_inv)


def test_base_add_reparameterisation_missing_requirements(reparam):
    """Assert an error is raised if a required parameter is missing"""
    r = MagicMock(spec=Reparameterisation)
    r.requires = ["x"]

    reparam.parameters = ["y"]
    reparam.prime_parameters = ["y_prime"]

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
    new_reparam = MagicMock(spec=Reparameterisation)
    with patch(
        "nessai.reparameterisations.combined.sort_reparameterisations",
        return_value=[new_reparam],
    ) as mock:
        CombinedReparameterisation.add_reparameterisations(
            reparam, new_reparam
        )
    reparam._add_reparameterisation.assert_called_once_with(new_reparam)
    mock.assert_called_once_with([new_reparam], existing_parameters=[])


def test_add_reparameterisations_multiple(reparam):
    """Assert add_reparameterisations works correctly with multiple reparams.

    Should call `_add_reparmeterisation` with each reparam.
    """
    parameters = ["x"]
    reparam.parameters = parameters
    r1 = MagicMock(spec=Reparameterisation)
    r2 = MagicMock(spec=Reparameterisation)
    reparams = [r1, r2]
    with patch(
        "nessai.reparameterisations.combined.sort_reparameterisations",
        return_value=[r1, r2],
    ) as mock:
        CombinedReparameterisation.add_reparameterisations(reparam, reparams)
    reparam._add_reparameterisation.assert_has_calls([call(r1), call(r2)])
    mock.assert_called_once_with(reparams, existing_parameters=parameters)


def test_check_order_valid(reparam, LightReparam):
    """Assert no error is raised if the order is valid"""
    d = dict(
        a=LightReparam("a", parameters=["x"], requires=[]),
        b=LightReparam("b", parameters=["y"], requires=["x"]),
        c=LightReparam("c", parameters=["z"], requires=["y"]),
    )
    reparam.__getitem__.side_effect = d.__getitem__
    reparam.from_prime_order = ["a", "b", "c"]

    CombinedReparameterisation.check_order(reparam)


def test_check_order_invalid(reparam, LightReparam):
    """Assert an error is raised if the order is invalid"""
    d = dict(
        a=LightReparam("a", parameters=["x"], requires=["z"]),
        b=LightReparam("b", parameters=["y"], requires=[]),
        c=LightReparam("c", parameters=["z"], requires=["y"]),
    )
    reparam.__getitem__.side_effect = d.__getitem__
    reparam.from_prime_order = ["a", "b", "c"]
    print(reparam["a"])

    with pytest.raises(
        RuntimeError, match="Order of reparameterisations is invalid *."
    ):
        CombinedReparameterisation.check_order(reparam)


@pytest.mark.parametrize("has", [False, True])
def test_has_prime_prior(reparam, has):
    """Assert correct value is returned for has_prime_prior.

    If any of the reparam are missing the prime prior should be False. If all
    of them have it, should be True.
    """
    r1 = MagicMock(spec=Reparameterisation)
    r2 = MagicMock(spec=Reparameterisation)
    r1.has_prime_prior = True
    r2.has_prime_prior = has
    reparam.values = MagicMock(return_value=[r1, r2])

    out = CombinedReparameterisation.has_prime_prior.__get__(reparam)
    assert out is has


@pytest.mark.parametrize("require", [False, True])
def test_requires_prime_prior(reparam, require):
    """Assert correct value is returned for requires prime prior.

    Should be True if any reparam requires the prime prior and False only if
    none of them do.
    """
    r1 = MagicMock(spec=Reparameterisation)
    r2 = MagicMock(spec=Reparameterisation)
    r1.requires_prime_prior = False
    r2.requires_prime_prior = require
    reparam.values = MagicMock(return_value=[r1, r2])

    out = CombinedReparameterisation.requires_prime_prior.__get__(reparam)
    assert out is require


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


def test_x_prime_log_prior(reparam):
    """Assert x_prime_log_prior is called for all reparams"""
    r1 = MagicMock(spec=Reparameterisation)
    r2 = MagicMock(spec=Reparameterisation)
    r1.x_prime_log_prior = MagicMock(return_value=-4)
    r2.x_prime_log_prior = MagicMock(return_value=-5)
    reparam.values = MagicMock(return_value=[r1, r2])

    x = np.array([(1, 2)], dtype=[("x", "f8"), ("y", "f8")])

    out = CombinedReparameterisation.x_prime_log_prior(reparam, x)

    r1.x_prime_log_prior.assert_called_once_with(x)
    r2.x_prime_log_prior.assert_called_once_with(x)
    assert out == -9
