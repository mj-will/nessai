
import pytest

from nessai.proposal import FlowProposal


@pytest.mark.parametrize("kwargs", [
    None, {""}
    ])
def test_flowproposal_init(tmpdir, model, kwargs):
    """Test the init function with defaults args"""
    output = str(tmpdir.mkdir('flowproposal'))
    fp = FlowProposal(model, output=output)
    fp.initialise()
