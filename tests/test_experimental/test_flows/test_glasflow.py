import numpy as np
import pytest

from nessai.experimental.flows.glasflow import (
    GlasflowWrapper,
    get_glasflow_class,
    known_flows,
)
from nessai.flowmodel import FlowModel


@pytest.mark.parametrize("name", known_flows.keys())
def test_get_glasflow_class(name):
    FlowClass = get_glasflow_class(f"glasflow-{name}")
    FlowClass(n_inputs=2, n_neurons=4, n_blocks=2, n_layers=1)


def test_get_glasflow_class_missing_prefix():
    with pytest.raises(ValueError, match=r"'glasflow' missing from name"):
        get_glasflow_class("realnvp")


def test_get_glasflow_class_invalid_flow():
    with pytest.raises(
        ValueError, match=r"invalid is not a known glasflow flow"
    ):
        get_glasflow_class("glasflow.invalid")


@pytest.mark.integration_test
def test_glasflow_integration(tmp_path):
    from glasflow.flows import RealNVP

    flow_config = dict(
        ftype="glasflow-realnvp",
        n_inputs=2,
    )

    flowmodel = FlowModel(flow_config=flow_config, output=tmp_path / "test")

    flowmodel.initialise()

    assert isinstance(flowmodel.model, GlasflowWrapper)
    assert isinstance(flowmodel.model._flow, RealNVP)

    flowmodel.train(np.random.randn(100, 2))
