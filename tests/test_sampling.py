import os

from nessai.flowsampler import FlowSampler
import torch
import pytest

torch.set_num_threads(1)


def test_sampling_with_rescale(model, flow_config, tmpdir):
    """
    Test sampling with rescaling. Checks that flow is trained.
    """
    output = str(tmpdir.mkdir('w_rescale'))
    fp = FlowSampler(model, output=output, resume=False, nlive=100, plot=False,
                     flow_config=flow_config, training_frequency=10,
                     maximum_uninformed=9, rescale_parameters=True,
                     seed=1234, max_iteration=11, poolsize=10, max_threads=1)
    fp.run()
    assert fp.ns.proposal.flow.weights_file is not None
    assert fp.ns.proposal.training_count == 1


def test_sampling_without_rescale(model, flow_config, tmpdir):
    """
    Test sampling without rescaling. Checks that flow is trained.
    """
    output = str(tmpdir.mkdir('wo_rescale'))
    fp = FlowSampler(model, output=output, resume=False, nlive=100, plot=False,
                     flow_config=flow_config, training_frequency=10,
                     maximum_uninformed=9, rescale_parameters=False, seed=1234,
                     max_iteration=11, poolsize=10)
    fp.run()
    assert fp.ns.proposal.flow.weights_file is not None
    assert fp.ns.proposal.training_count == 1


def test_sampling_with_maf(model, flow_config, tmpdir):
    """
    Test sampling with MAF. Checks that flow is trained but does not
    check convergence.
    """
    flow_config['model_config']['ftype'] = 'maf'
    output = str(tmpdir.mkdir('maf'))
    fp = FlowSampler(model, output=output, resume=False, nlive=100, plot=False,
                     flow_config=flow_config, training_frequency=10,
                     maximum_uninformed=9, rescale_parameters=True,
                     seed=1234, max_iteration=11, poolsize=10)
    fp.run()
    assert fp.ns.proposal.flow.weights_file is not None
    assert fp.ns.proposal.training_count == 1


@pytest.mark.parametrize('analytic', [False, True])
def test_sampling_uninformed(model, flow_config, tmpdir, analytic):
    """
    Test running the sampler with the two uninformed proposal methods.
    """
    output = str(tmpdir.mkdir('uninformed'))
    fp = FlowSampler(model, output=output, resume=False, nlive=100, plot=False,
                     flow_config=flow_config, training_frequency=None,
                     maximum_uninformed=10, rescale_parameters=True,
                     seed=1234, max_iteration=11, poolsize=10,
                     analytic_proposal=analytic)
    fp.run()


def test_sampling_with_n_pool(model, flow_config, tmpdir):
    """
    Test running the sampler with multiprocessing.
    """
    output = str(tmpdir.mkdir('pool'))
    fp = FlowSampler(model, output=output, resume=False, nlive=100, plot=False,
                     flow_config=flow_config, training_frequency=10,
                     maximum_uninformed=9, rescale_parameters=True,
                     seed=1234, max_iteration=11, poolsize=10, max_threads=3,
                     n_pool=2)
    fp.run()
    assert fp.ns.proposal.flow.weights_file is not None
    assert fp.ns.proposal.training_count == 1
    assert os.path.exists(output + '/result.json')
