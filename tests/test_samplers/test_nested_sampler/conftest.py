import pytest
from unittest.mock import create_autospec

from nessai.samplers.nestedsampler import NestedSampler


@pytest.fixture(scope="function")
def sampler(model):
    s = create_autospec(NestedSampler)
    s.nlive = 100
    s.model = model
    s.store_live_points = False
    return s
