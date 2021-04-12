# -*- coding: utf-8 -*-
"""
Tests related to using the flow.
"""
import pytest
from unittest.mock import MagicMock

from nessai.nestedsampler import NestedSampler


@pytest.mark.parametrize('switch', [False, True])
@pytest.mark.parametrize('uninformed', [False, True])
def test_check_state_force(sampler, switch, uninformed):
    """Test the behaviour of check_state with force=True.

    Training should always start irrespective of other checks and with
    force=True unless uninformed sampling is being used and the switch=False.
    """
    sampler.uninformed_sampling = False
    sampler.check_proposal_switch = MagicMock(return_value=switch)
    sampler.check_training = MagicMock()
    sampler.train_proposal = MagicMock()

    NestedSampler.check_state(sampler, force=True)

    if uninformed and not switch:
        sampler.train_proposal.assert_not_called
    else:
        sampler.train_proposal.assert_called_once_with(force=True)

    sampler.check_training.assert_not_called
