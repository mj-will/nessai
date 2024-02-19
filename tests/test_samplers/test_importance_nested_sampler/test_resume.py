import datetime
import numpy as np
import pickle
import pytest
from unittest.mock import MagicMock, patch

from nessai.samplers.importancesampler import ImportanceNestedSampler as INS


def test_getstate_no_model(ins):
    ins.proposal = MagicMock()
    ins.model = None
    state, proposal, training_samples, iid_samples = INS.__getstate__(ins)
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
    assert "model" not in state
    assert state["_previous_likelihood_evaluations"] == evals
    assert state["_previous_likelihood_evaluation_time"] == 30
    assert proposal is ins.proposal
    assert training_samples is ins.training_samples
    assert iid_samples is ins.iid_samples


@pytest.mark.parametrize("has_log_q", [False, True])
def test_resume_from_pickled_sampler(model, samples, has_log_q):

    sampler = MagicMock()

    obj = MagicMock()
    obj.log_q = None
    obj.log_evidence = 0.0
    obj.log_evidence_error = 1.0
    obj.proposal = MagicMock()
    log_meta_proposal = np.log(np.random.rand(len(samples)))
    log_q = np.log(np.random.rand(len(samples)))
    log_meta_proposal_iid = np.log(np.random.rand(len(samples)))
    log_q_iid = np.log(np.random.rand(len(samples)))
    obj.proposal.compute_meta_proposal_samples = MagicMock(
        side_effect=[
            (log_meta_proposal, log_q),
            (log_meta_proposal_iid, log_q_iid),
        ]
    )
    obj.training_samples.samples = samples
    obj.iid_samples.samples = samples
    if has_log_q:
        obj.training_samples.log_q = log_q
        obj.iid_samples.log_q = log_q_iid
    else:
        obj.training_samples.log_q = None
        obj.iid_samples.log_q = None

    with patch(
        "nessai.samplers.importancesampler.BaseNestedSampler.resume_from_pickled_sampler",  # noqa
        return_value=obj,
    ) as mock_resume:
        out = INS.resume_from_pickled_sampler(sampler, model)

    mock_resume.assert_called_once_with(sampler, model)
    if has_log_q:
        obj.proposal.compute_meta_proposal_samples.assert_not_called()
    else:
        obj.proposal.compute_meta_proposal_samples.assert_called()

    assert out.training_samples.log_q is log_q
    assert out.iid_samples.log_q is log_q_iid


@pytest.mark.parametrize("save_log_q", [True, False])
@pytest.mark.integration_test
def test_pickling_sampler_integration(integration_model, tmp_path, save_log_q):
    outdir = tmp_path / "test_pickle"
    ins = INS(
        model=integration_model,
        output=outdir,
        nlive=50,
        min_samples=10,
        max_iteration=1,
        save_log_q=save_log_q,
        plot=False,
        checkpointing=False,
    )
    ins.nested_sampling_loop()
    data = pickle.dumps(ins)
    loaded_ins = pickle.loads(data)
    if save_log_q:
        np.testing.assert_array_equal(
            loaded_ins._ordered_samples.log_q, ins._ordered_samples.log_q
        )
    else:
        assert loaded_ins._ordered_samples.log_q is None
