from unittest.mock import create_autospec

import pytest

from nessai.samplers.nestedsampler import NestedSampler


@pytest.fixture(scope="function")
def sampler(model, rng):
    s = create_autospec(NestedSampler)
    s.nlive = 100
    s.model = model
    s.store_live_points = False
    s.rng = rng
    return s
