# -*- coding: utf-8 -*-
"""
Test the analytic proposal method.
"""
import pytest

from nessai.proposal import AnalyticProposal


@pytest.fixture(scope='function')
def proposal(model):
    return AnalyticProposal(model)


def test_populate(proposal):
    """Test the population process"""
    N = 500
    proposal.populate(N=N)

    assert proposal.samples.size == N
    assert proposal.populated


def test_draw(model, proposal):
    """Test the draw method"""
    N = 10
    old_point = model.new_point()
    assert not proposal.populated
    proposal.draw(old_point, N=N)
    assert proposal.populated
    [proposal.draw(old_point) for _ in range(N - 1)]
    assert not proposal.indices
    assert not proposal.populated
