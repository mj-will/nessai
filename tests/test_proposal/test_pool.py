# -*- coding: utf-8 -*-
"""
Test the multiprocessing pool for the different proposal methods.
"""
from numpy.testing import assert_array_equal
import pytest

from nessai.proposal import (
    AnalyticProposal,
    AugmentedFlowProposal,
    FlowProposal,
    RejectionProposal
)


@pytest.mark.parametrize(
    'proposal',
    [AnalyticProposal, AugmentedFlowProposal, FlowProposal, RejectionProposal]
)
@pytest.mark.parametrize('n_pool', [None, 1])
@pytest.mark.timeout(10)
@pytest.mark.flaky(run=3)
@pytest.mark.integration_test
def test_analytic_pool(proposal, n_pool, model, tmpdir):
    try:
        prop = proposal(model, n_pool=n_pool, poolsize=2, plot=False,
                        max_radius=1, output=str(tmpdir.mkdir('test')))
    except TypeError:
        prop = proposal(model, n_pool=n_pool, poolsize=2)

    prop.initialise()

    prop.configure_pool()
    if n_pool:
        assert prop.pool
    else:
        assert prop.pool is None

    worst = model.new_point()
    prop.draw(worst)

    if n_pool:
        logL = [model.log_likelihood(s) for s in prop.samples]
        assert_array_equal(prop.samples['logL'], logL)
    else:
        assert all(ll == 0. for ll in prop.samples['logL'])

    prop.close_pool()
    assert prop.pool is None
