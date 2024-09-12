import pytest

from nessai.experimental.proposal.clustering import ClusteringFlowProposal
from nessai.flowsampler import FlowSampler


@pytest.mark.requires("faiss")
@pytest.mark.slow_integration_test
def test_sampling_with_clusteringflowproposal(integration_model, tmp_path):
    fp = FlowSampler(
        integration_model,
        output=tmp_path / "clustering_sampling_test",
        resume=False,
        nlive=100,
        plot=False,
        maximum_uninformed=100,
        seed=1234,
        max_iteration=200,
        flow_class="clusteringflowproposal",
    )
    fp.run()
    assert isinstance(fp.ns.proposal, ClusteringFlowProposal)
