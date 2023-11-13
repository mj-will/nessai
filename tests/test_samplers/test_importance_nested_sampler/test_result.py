"""Tests related to the results returned by the sampler"""
import datetime
from unittest.mock import MagicMock

from nessai.samplers.importancesampler import ImportanceNestedSampler as INS
from nessai.evidence import _INSIntegralState
import numpy as np


def test_get_result_dictionary(ins, history, samples):
    ins.samples = samples.copy()
    ins.final_samples = samples
    ins.history = history
    ins.state = MagicMock(spec=_INSIntegralState)

    ins.seed = 1234
    ins.sampling_time = datetime.timedelta(seconds=10)
    ins.training_time = datetime.timedelta(seconds=10)
    ins.draw_samples_time = datetime.timedelta(seconds=10)
    ins.add_and_update_samples_time = datetime.timedelta(seconds=10)
    ins.draw_final_samples_time = datetime.timedelta(seconds=10)

    ins.initial_log_evidence = -1.0
    ins.initial_log_evidence_error = 0.1

    ins.log_evidence = -1.1
    ins.log_evidence_error = 0.3

    ins.bootstrap_log_evidence = None
    ins.bootstrap_log_evidence_error = None

    ins.state.log_posterior_weights = np.log(np.random.rand(len(samples)))

    ins.importance = {
        "total": [1, 2, 3],
        "evidence": [1, 2, 3],
        "posterior": [1, 2, 3],
    }

    out = INS.get_result_dictionary(ins)

    assert isinstance(out, dict)
