import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from nessai.samplers.importancesampler import ImportanceNestedSampler as INS


def test_getstate_no_model(ins):
    ins.proposal = MagicMock()
    ins.model = None
    state, proposal, training_samples, iid_samples = INS.__getstate__(ins)
    assert state["log_q"] is None
    assert "model" not in state
    assert state["_previous_likelihood_evaluations"] == 0
    assert state["_previous_likelihood_evaluation_time"] == 0
    assert proposal is ins.proposal
    assert training_samples is ins.training_samples
    assert iid_samples is ins.iid_samples


def test_getstate_model(ins):

    evals = 10
    time = datetime.timedelta(seconds=30)

    ins.proposal = MagicMock()
    ins.model = MagicMock()
    ins.model.likelihood_evaluations = evals
    ins.model.likelihood_evaluation_time = time

    state, proposal, training_samples, iid_samples = INS.__getstate__(ins)
    assert state["log_q"] is None
    assert "model" not in state
    assert state["_previous_likelihood_evaluations"] == evals
    assert state["_previous_likelihood_evaluation_time"] == 30
    assert proposal is ins.proposal
    assert training_samples is ins.training_samples
    assert iid_samples is ins.iid_samples


def test_resume_from_pickled_sampler(model, samples):

    sampler = MagicMock()

    obj = MagicMock()
    obj.log_q = None
    obj.log_evidence = 0.0
    obj.log_evidence_error = 1.0
    obj.proposal = MagicMock()
    obj.training_samples.samples = samples
    log_meta_proposal = np.log(np.random.rand(len(samples)))
    log_q = np.log(np.random.rand(len(samples)))
    obj.proposal.compute_meta_proposal_samples = MagicMock(
        return_value=(log_meta_proposal, log_q)
    )

    with patch(
        "nessai.samplers.importancesampler.BaseNestedSampler.resume_from_pickled_sampler",  # noqa
        return_value=obj,
    ) as mock_resume:
        out = INS.resume_from_pickled_sampler(sampler, model)

    mock_resume.assert_called_once_with(sampler, model)

    assert out.training_samples.log_q is log_q
