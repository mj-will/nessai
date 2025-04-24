from unittest.mock import MagicMock

from nessai.samplers.importancesampler import ImportanceNestedSampler as INS


def test_update_history(ins, history):
    ins.history = history
    ins.log_likelihood_threshold = -1.23
    ins.logX = -0.5
    ins.gradient = 0.9
    ins.samples_entropy = 1.1
    ins.current_proposal_entropy = 0.8
    ins.live_points_ess = 300

    ins.stopping_criteria = ["ess", "ratio"]
    ins.state = MagicMock()
    ins.state.ess = 500
    ins.state.ratio = 1.0

    INS.update_history(ins)

    assert ins.history["logL_threshold"][-1] == -1.23
    assert ins.history["logX"][-1] == -0.5
    assert ins.history["gradients"][-1] == 0.9
    assert ins.history["samples_entropy"][-1] == 1.1
    assert ins.history["proposal_entropy"][-1] == 0.8
    assert ins.history["live_points_ess"][-1] == 300
    assert ins.history["stopping_criteria"]["ess"][-1] == 500
    assert ins.history["stopping_criteria"]["ratio"][-1] == 1.0
