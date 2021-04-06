# -*- coding: utf-8 -*-
"""
Test functions related to training and using the flow.
"""
from nessai.proposal import FlowProposal
import numpy as np
from unittest.mock import MagicMock, patch


def test_reset_model_weights(proposal):
    """Test reseting model weights"""
    proposal.flow = MagicMock()
    proposal.flow.reset_model = MagicMock()
    FlowProposal.reset_model_weights(proposal, reset_permutations=True)
    proposal.flow.reset_model.assert_called_once_with(reset_permutations=True)


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_train_plot_false(mock_os_makedirs, proposal, model):
    """Test the train method"""
    x = model.new_point(2)
    x_prime = model.new_point(2)
    proposal.rescaled_names = model.names
    proposal.save_training_data = False
    proposal.training_count = 0
    proposal.populated = True
    proposal.flow = MagicMock()
    proposal.flow.train = MagicMock()
    proposal.check_state = MagicMock()
    proposal.rescale = MagicMock(return_value=(x_prime, np.zeros_like(x)))
    FlowProposal.train(proposal, x, plot=False)

    assert np.array_equal(x, proposal.training_data)
    proposal.check_state.assert_called_once_with(proposal.training_data)
    proposal.rescale.assert_called_once_with(x)
    assert proposal.populated is False
    assert proposal.training_count == 1
    proposal.flow.train.assert_called_once()
