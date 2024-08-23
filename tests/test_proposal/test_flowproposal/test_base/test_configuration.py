import pytest

from nessai.proposal.flowproposal.base import BaseFlowProposal


def test_conifgure_poolsize(proposal):
    BaseFlowProposal.configure_poolsize(proposal, 10, True, 10)
    assert proposal._poolsize == 10
    assert proposal._poolsize_scale == 1
    assert proposal.update_poolsize is True
    assert proposal.max_poolsize_scale == 10
    assert proposal.ns_acceptance == 1.0


def test_config_poolsize_none(proposal):
    """
    Test the popluation configuration raises an error if poolsize is None.
    """
    with pytest.raises(RuntimeError, match=r"Must specify `poolsize`"):
        BaseFlowProposal.configure_poolsize(
            proposal,
            None,
            True,
            10,
        )


@pytest.mark.parametrize(
    "plot, plot_pool, plot_train",
    [
        (True, True, True),
        ("all", "all", "all"),
        ("train", False, "all"),
        ("pool", "all", False),
        ("min", True, True),
        ("minimal", True, True),
        (False, False, False),
        ("some", False, False),
    ],
)
def test_configure_plotting(proposal, plot, plot_pool, plot_train):
    """Test the configuration of plotting settings"""
    BaseFlowProposal.configure_plotting(proposal, plot)
    assert proposal._plot_pool == plot_pool
    assert proposal._plot_training == plot_train


def test_update_flow_proposal(proposal):
    """Assert the number of inputs is updated"""
    proposal.flow_config = {"model_config": {}}
    proposal.rescaled_dims = 4
    BaseFlowProposal.update_flow_config(proposal)
    assert proposal.flow_config["n_inputs"] == 4


def test_flow_config(proposal):
    """Assert the correct config is returned"""
    config = {"a": 1}
    proposal._flow_config = config
    assert BaseFlowProposal.flow_config.__get__(proposal) is config
