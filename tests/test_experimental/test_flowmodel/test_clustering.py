from unittest.mock import create_autospec, patch

import numpy as np
import pytest

from nessai.experimental.flowmodel.clustering import ClusteringFlowModel as CFM


@pytest.fixture
def cfm():
    return create_autospec(CFM)


@pytest.mark.requires("faiss")
def test_init(cfm, tmp_path, caplog, rng):
    caplog.set_level("DEBUG")
    flow_config = {}
    training_config = {}
    output = tmp_path
    with patch(
        "nessai.experimental.flowmodel.clustering.FlowModel.__init__"
    ) as mock_parent_init:
        CFM.__init__(
            cfm,
            flow_config=flow_config,
            training_config=training_config,
            output=output,
            rng=rng,
        )

    mock_parent_init.assert_called_once_with(
        flow_config=flow_config,
        training_config=training_config,
        output=output,
        rng=rng,
    )
    assert "faiss version" in str(caplog.text)


@pytest.mark.requires("faiss")
@pytest.mark.integration_test
def test_clustering_integration(tmp_path, caplog):
    caplog.set_level("DEBUG")
    fm = CFM(flow_config=dict(n_inputs=2), output=tmp_path)
    n = 100
    samples = np.concatenate(
        [
            -10 * np.ones([n, 2]),
            10 * np.ones([n, 2]),
        ],
        axis=0,
    )
    fm.train_clustering(samples)
    assert fm.n_clusters == 2

    labels_pos = fm.get_cluster_labels(np.array([[10, 10]]))
    labels_neg = fm.get_cluster_labels(np.array([[-10, -10]]))
    assert labels_neg != labels_pos
