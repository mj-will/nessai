
from nessai.flowsampler import FlowSampler
import torch

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
