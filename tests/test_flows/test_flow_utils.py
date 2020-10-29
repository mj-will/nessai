
from nflows.transforms.permutations import RandomPermutation
import torch

from flowproposal.flows.utils import weight_reset


def test_weight_reset_permutation():
    """Test to make sure random permutation is reset correctly"""
    x = torch.arange(10).reshape(1, -1)
    m = RandomPermutation(features=10)
    y_init, _ = m(x)
    p = m._permutation.numpy()
    m.apply(weight_reset)
    y_reset, _ = m(x)
    assert not (p == m._permutation.numpy()).all()
    assert not (y_init.numpy() == y_reset.numpy()).all()
