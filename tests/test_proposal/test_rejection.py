# -*- coding: utf-8 -*-
"""
Test the rejection proposal class.
"""
import pytest

from nessai.proposal import RejectionProposal


@pytest.fixture()
def poolsize():
    return 10


@pytest.fixture(scope='function')
def proposal(model, poolsize):
    return RejectionProposal(model, poolsize=poolsize)


def test_populate(proposal, poolsize):
    """Test the population process"""
    proposal.populate(N=poolsize)

    assert proposal.samples.size == poolsize
    assert proposal.populated


def test_draw(model, proposal, poolsize):
    """Test the draw method"""
    old_point = model.new_point()
    assert not proposal.populated
    proposal.draw(old_point)
    assert proposal.populated
    [proposal.draw(old_point) for _ in range(poolsize - 1)]
    assert not proposal.indices
    assert not proposal.populated
