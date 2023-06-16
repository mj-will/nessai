from unittest.mock import create_autospec

import pytest
from nessai.samplers.importancesampler import ImportanceNestedSampler
from nessai.livepoint import (
    numpy_array_to_live_points,
    add_extra_parameters_to_live_points,
    reset_extra_live_points_parameters,
)
import numpy as np


@pytest.fixture(scope="module", autouse=True)
def ins_livepoint_params():
    reset_extra_live_points_parameters()
    add_extra_parameters_to_live_points(["logW", "logQ"])
    # Test happens here
    yield

    # Called after every test
    reset_extra_live_points_parameters()


@pytest.fixture
def ins():
    return create_autospec(ImportanceNestedSampler)


@pytest.fixture
def samples(model):
    x = numpy_array_to_live_points(np.random.randn(1000, 2), model.names)
    x["it"] = np.random.randint(0, 10, size=len(x))
    x["logL"] = model.log_likelihood(x)
    x["logP"] = model.log_prior(x)
    return x
