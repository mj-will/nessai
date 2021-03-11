
import pytest

from nessai.nestedsampler import NestedSampler


@pytest.fixture(scope='function')
def sampler(model, tmpdir):
    output = str(tmpdir.mkdir('test'))
    return NestedSampler(model, nlive=10, output=output)
