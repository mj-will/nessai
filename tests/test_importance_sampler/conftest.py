# -*- coding: utf-8 -*-
"""
Config for importance sampler tests.
"""
import numpy as np
import pytest
from unittest.mock import create_autospec

from nessai.importancesampler import ImportanceNestedSampler
from nessai.livepoint import (
    add_extra_parameters_to_live_points,
    numpy_array_to_live_points,
    reset_extra_live_points_parameters,
)


@pytest.fixture(scope="module", autouse=True)
def extra_fields():
    """Fixture to add extra fields to the live points dtype"""
    add_extra_parameters_to_live_points(['logW', 'logQ'])
    yield
    reset_extra_live_points_parameters()


@pytest.fixture
def sampler() -> ImportanceNestedSampler:
    return create_autospec(ImportanceNestedSampler)


@pytest.fixture
def live_points():
    return numpy_array_to_live_points(
        np.arange(10, 20, 1)[:, np.newaxis],
        names=['x']
    )


@pytest.fixture
def nested_samples():
    return numpy_array_to_live_points(
        np.arange(10)[:, np.newaxis],
        names=['x']
    )
