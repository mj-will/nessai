"""Configuration for ImportanceFlowProposal tests"""

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest

from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model
from nessai.proposal.importance import ImportanceFlowProposal

NSAMPLES = 10


@pytest.fixture()
def ifp(rng):
    obj = create_autospec(ImportanceFlowProposal, rng=rng)
    obj.model = MagicMock(spec=Model)
    return obj


@pytest.fixture()
def x_array(model):
    return np.random.randn(NSAMPLES, model.dims)


@pytest.fixture
def x(x_array, model):
    return numpy_array_to_live_points(x_array, model.names)


@pytest.fixture()
def x_prime(model):
    return np.random.randn(NSAMPLES, model.dims)


@pytest.fixture
def log_j():
    return np.random.randn(NSAMPLES)
