"""Configuration for ImportanceFlowProposal tests"""

import pytest
from unittest.mock import MagicMock, create_autospec

from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model
from nessai.proposal.importance import ImportanceFlowProposal
import numpy as np

NSAMPLES = 10


@pytest.fixture()
def ifp():
    obj = create_autospec(ImportanceFlowProposal)
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
