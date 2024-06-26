from nessai.experimental.flowmodel.clustering import ClusteringFlowModel as CFM
import numpy as np
import pytest
from unittest.mock import create_autospec, patch


@pytest.fixture
def cfm():
    return create_autospec(CFM)


def test_init(cfm, tmp_path, caplog):
    caplog.set_level("DEBUG")
    config = {}
    output = tmp_path
    with patch(
        "nessai.experimental.flowmodel.clustering.FlowModel.__init__"
    ) as mock_parent_init:
        CFM.__init__(cfm, config=config, output=output)

    mock_parent_init.assert_called_once_with(config=config, output=output)
    assert "faiss version" in str(caplog.text)


@pytest.mark.integration_test
def test_clustering_integration(tmp_path, caplog):
    caplog.set_level("DEBUG")
    fm = CFM(config=dict(model_config=dict(n_inputs=2)), output=tmp_path)
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
