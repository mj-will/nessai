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


@pytest.mark.parametrize(
    "stopping_criterion, tolerance, check_criteria",
    [
        ("ess", 1000, "all"),
        (["ess", "log_evidence_ratio"], [1000, 0], "all"),
        (["ess", "log_evidence_ratio"], [1000, 0], "any"),
    ],
)
def test_configure_stopping_criterion(
    ins, stopping_criterion, tolerance, check_criteria
):
    INS.configure_stopping_criterion(
        ins, stopping_criterion, tolerance, check_criteria
    )

    if isinstance(stopping_criterion, str):
        stopping_criterion = [stopping_criterion]

    if isinstance(tolerance, (int, float)):
        tolerance = [float(tolerance)]
    else:
        tolerance = [float(t) for t in tolerance]

    assert ins.combined_criterion.names == stopping_criterion
    assert list(ins.combined_criterion.tolerances.values()) == tolerance

    if check_criteria == "all":
        assert ins.combined_criterion.mode == "and"
    elif check_criteria == "any":
        assert ins.combined_criterion.mode == "or"
    else:
        raise ValueError("check_criteria must be 'all' or 'any'")


def test_stopping_criteria(ins):
    ins.combined_criterion = MagicMock()
    ins.combined_criterion.names = ["ratio", "ess"]
    INS.stopping_criteria.__get__(ins) == ins.combined_criterion.names
