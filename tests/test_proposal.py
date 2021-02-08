import numpy as np
import pytest
import torch

from nessai.proposal import FlowProposal
from nessai.livepoint import numpy_array_to_live_points

torch.set_num_threads(1)


@pytest.mark.parametrize('latent_prior', ['gaussian', 'truncated_gaussian',
                                          'uniform_nball', 'uniform_nsphere',
                                          'uniform'])
@pytest.mark.parametrize('expansion_fraction', [0, 1, None])
@pytest.mark.parametrize('check_acceptance', [False, True])
@pytest.mark.parametrize('rescale_parameters', [False, True])
@pytest.mark.parametrize('max_radius', [False, 2])
@pytest.mark.timeout(10)
@pytest.mark.flaky(run=3)
def test_flowproposal_populate(tmpdir, model, latent_prior, expansion_fraction,
                               check_acceptance, rescale_parameters,
                               max_radius):
    """
    Test the populate method in the FlowProposal class with a range of
    parameters
    """
    output = str(tmpdir.mkdir('flowproposal'))
    fp = FlowProposal(
        model,
        output=output,
        plot=False,
        poolsize=1000,
        latent_prior=latent_prior,
        expansion_fraction=expansion_fraction,
        check_acceptance=check_acceptance,
        rescale_parameters=rescale_parameters,
        max_radius=max_radius
    )

    fp.initialise()
    worst = numpy_array_to_live_points(0.5 * np.ones(fp.dims), fp.names)
    fp.populate(worst, N=100)

    assert fp.x.size == 100
