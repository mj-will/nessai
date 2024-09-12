from unittest.mock import MagicMock

import pytest

from nessai.samplers.importancesampler import ImportanceNestedSampler as INS


@pytest.mark.parametrize(
    "criterion, tolerance, stop_any, reached",
    [
        ([5.0], [0.0], False, False),
        ([-1.0], [0.0], False, True),
        ([5.0, 1.5], [0.0, 1], False, False),
        ([5.0, 0.5], [0.0, 1], False, False),
        ([-1.0, 0.5], [0.0, 1], False, True),
        ([5.0, 1.5], [0.0, 1], True, False),
        ([5.0, 0.5], [0.0, 1], True, True),
        ([-1.0, 0.5], [0.0, 1], True, True),
    ],
)
def test_reached_tolerance(ins, criterion, tolerance, stop_any, reached):
    ins.criterion = criterion
    ins.tolerance = tolerance
    ins._stop_any = stop_any
    assert INS.reached_tolerance.__get__(ins) is reached


def test_compute_stopping_criterion_fractional_error(ins):
    ins.stopping_criterion = ["fractional_error"]
    ins.iteration = 0
    ins.tolerance = 0.0
    ins.state = MagicMock()
    ins.state.evidence = 1.0
    ins.state.evidence_error = 0.1
    assert INS.compute_stopping_criterion(ins) == [0.1]
