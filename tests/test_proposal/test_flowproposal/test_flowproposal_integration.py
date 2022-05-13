# -*- coding: utf-8 -*-
"""End-to-end test of the populate method"""
import numpy as np
import pytest
from scipy import stats
import torch

from nessai.proposal import FlowProposal
from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model

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
@pytest.mark.integration_test
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
        poolsize=100,
        latent_prior=latent_prior,
        expansion_fraction=expansion_fraction,
        check_acceptance=check_acceptance,
        rescale_parameters=rescale_parameters,
        max_radius=max_radius,
        constant_volume_mode=False,
    )

    fp.initialise()
    worst = numpy_array_to_live_points(0.5 * np.ones(fp.dims), fp.names)
    fp.populate(worst, N=100)

    assert fp.x.size == 100


@pytest.mark.parametrize('plot', [False, True])
@pytest.mark.integration_test
def test_training(tmpdir, model, plot):
    """Integration test to test training the flow with and without plotting."""
    output = str(tmpdir.mkdir('test_train'))
    config = dict(max_epochs=10)
    fp = FlowProposal(
        model,
        output=output,
        plot='min',
        poolsize=100,
        flow_config=config)

    fp.initialise()

    x = model.new_point(500)
    fp.train(x, plot=plot)

    assert fp.training_count == 1
    assert fp.populated is False


@pytest.mark.parametrize('check_acceptance', [False, True])
@pytest.mark.parametrize('rescale_parameters', [False, True])
@pytest.mark.timeout(10)
@pytest.mark.flaky(run=3)
@pytest.mark.integration_test
def test_constant_volume_mode(
    tmpdir, model, check_acceptance, rescale_parameters
):
    """Integration test for constant volume mode.

    With q=0.8647 should get a radius of ~2.
    """
    output = str(tmpdir.mkdir('flowproposal'))
    expected_radius = 2.0
    fp = FlowProposal(
        model,
        output=output,
        plot=False,
        poolsize=10,
        constant_volume_mode=True,
        volume_fraction=0.8647,
        check_acceptance=check_acceptance,
        rescale_parameters=rescale_parameters,
    )
    fp.initialise()
    worst = numpy_array_to_live_points(0.5 * np.ones(fp.dims), fp.names)
    fp.populate(worst, N=100)
    assert fp.x.size == 100

    np.testing.assert_approx_equal(fp.r, expected_radius, 4)
    np.testing.assert_approx_equal(fp.fixed_radius, expected_radius, 4)


@pytest.mark.parametrize(
    "reparameterisations",
    [None, {'default': {'parameters': ['x', 'y']}}]
)
@pytest.mark.slow_integration_test
def test_prior_only_parameters(tmp_path, reparameterisations):
    """Integration test with prior only parameters"""
    output = tmp_path / "test_prior_only"
    output.mkdir()

    class TestModel(Model):

        def __init__(self):
            self.names = ['x', 'y', 'z']
            self.bounds = {n: [-10.0, 10.0] for n in self.names}

        def log_prior(self, x):
            log_p = np.log(self.in_bounds(x))
            for n in self.names:
                log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
            return log_p

        def log_likelihood(self, x):
            log_l = np.zeros(x.size)
            for n in ['x', 'y']:
                log_l += stats.norm.logpdf(x[n])
            return log_l

        def sample_parameter(self, p, n=1):
            return np.random.uniform(
                self.bounds[p][0],
                self.bounds[p][1],
                size=n
            )

    model = TestModel()

    fp = FlowProposal(
        model,
        output=output,
        poolsize=100,
        prior_only_parameters=['z'],
        constant_volume_mode=True,
        volume_fraction=0.5,
        reparameterisations=reparameterisations,
        rescale_parameters=True,
    )
    fp.initialise()

    training_data = model.new_point(500)
    fp.train(training_data)
    worst = numpy_array_to_live_points(0.5 * np.ones(fp.dims), fp.names)
    fp.populate(worst, N=100)
