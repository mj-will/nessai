from unittest.mock import MagicMock

import pytest

from nessai.samplers.importancesampler import ImportanceNestedSampler as INS


@pytest.mark.parametrize("met", [False, True])
def test_reached_tolerance(ins, met):
    ins.combined_criterion = MagicMock()
    ins.combined_criterion.is_met.return_value = met
    ins.criterion = {"ess": 1000, "ratio": -0.1}
    assert INS.reached_tolerance.__get__(ins) is met


def test_compute_stopping_criterion(ins):
    ins.combined_criterion = MagicMock()
    ins.combined_criterion.names = ["ratio", "ess"]
    ins.combined_criterion.tolerances = [0.0, 1000]
    ins.state = MagicMock()
    ins.state.ess = 500
    ins.state.ratio = 1.0
    assert INS.compute_stopping_criterion(ins) == {"ratio": 1.0, "ess": 500}
