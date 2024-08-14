# -*- coding: utf-8 -*-
"""Test different properties in FlowProposal"""
from nessai import config
from nessai.proposal import FlowProposal
import numpy as np


EXTRA_PARAMS_DTYPE = [
    (nsp, d)
    for nsp, d in zip(
        config.livepoints.non_sampling_parameters,
        config.livepoints.non_sampling_dtype,
    )
]


def test_poolsize(proposal):
    """Test poolsize property"""
    proposal._poolsize = 10
    proposal._poolsize_scale = 2
    assert FlowProposal.poolsize.__get__(proposal) == 20


def test_dims(proposal):
    """Test dims property"""
    proposal.parameters = ["x", "y"]
    assert FlowProposal.dims.__get__(proposal) == 2


def test_rescaled_dims(proposal):
    """Test rescaled_dims property"""
    proposal.prime_parameters = ["x", "y"]
    assert FlowProposal.rescaled_dims.__get__(proposal) == 2


def test_dtype(proposal):
    """Test dims property"""
    proposal.parameters = ["x", "y"]
    proposal._x_dtype = None
    assert (
        FlowProposal.x_dtype.__get__(proposal)
        == [("x", "f8"), ("y", "f8")] + EXTRA_PARAMS_DTYPE
    )


def test_prime_dtype(proposal):
    """Test dims property"""
    proposal.prime_parameters = ["x", "y"]
    proposal._x_prime_dtype = None
    assert (
        FlowProposal.x_prime_dtype.__get__(proposal)
        == [("x", "f8"), ("y", "f8")] + EXTRA_PARAMS_DTYPE
    )


def test_population_dtype(proposal):
    """Test dims property"""
    proposal.x_dtype = [
        ("x", "f8"),
        ("y", "f8"),
        ("logP", "f8"),
        ("logL", "f8"),
    ]
    proposal.use_x_prime_prior = False
    assert FlowProposal.population_dtype.__get__(proposal) == [
        ("x", "f8"),
        ("y", "f8"),
        ("logP", "f8"),
        ("logL", "f8"),
    ]


def test_population_dtype_prime_prior(proposal):
    """Test dims property"""
    proposal.x_prime_dtype = [
        ("x_p", "f8"),
        ("y_p", "f8"),
        ("logP", "f8"),
        ("logL", "f8"),
    ]
    proposal.use_x_prime_prior = True
    assert FlowProposal.population_dtype.__get__(proposal) == [
        ("x_p", "f8"),
        ("y_p", "f8"),
        ("logP", "f8"),
        ("logL", "f8"),
    ]


def test_prior_bounds_no_unit_hypercube(proposal, model):
    proposal._prior_bounds = None
    proposal.map_to_unit_hypercube = False
    proposal.model = model
    out = FlowProposal.prior_bounds.__get__(proposal)
    assert out is model.bounds
    assert proposal._prior_bounds is model.bounds


def test_prior_bounds_unit_hypercube(proposal, model):
    proposal._prior_bounds = None
    proposal.map_to_unit_hypercube = True
    proposal.model = model
    out = FlowProposal.prior_bounds.__get__(proposal)
    assert list(out.keys()) == model.names
    assert all(np.array_equal(v, np.array([0, 1])) for v in out.values())
    assert proposal._prior_bounds is out
