"""Tests related to the results returned by the sampler"""

import datetime
from unittest.mock import MagicMock, create_autospec

from nessai.samplers.importancesampler import (
    ImportanceNestedSampler as INS,
    OrderedSamples,
)
from nessai.evidence import _INSIntegralState
import numpy as np


def test_get_result_dictionary(ins, history, samples, iid):
    ins.training_samples.samples = samples.copy()
    ins.training_samples.state.log_evidence = -1.1
    ins.training_samples.state.log_evidence_error = 0.2
    if iid:
        ins.iid_samples = create_autospec(OrderedSamples)
        ins.iid_samples.state = create_autospec(_INSIntegralState)
        ins.iid_samples.samples = samples.copy()
        ins.iid_samples.state.log_evidence = -0.9
        ins.iid_samples.state.log_evidence_error = 0.14
    ins.final_samples = samples
    ins.history = history
    ins.state = MagicMock(spec=_INSIntegralState)

    ins.seed = 1234
    ins.sampling_time = datetime.timedelta(seconds=10)
    ins.training_time = datetime.timedelta(seconds=10)
    ins.draw_samples_time = datetime.timedelta(seconds=10)
    ins.add_and_update_samples_time = datetime.timedelta(seconds=10)
    ins.draw_final_samples_time = datetime.timedelta(seconds=10)

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
