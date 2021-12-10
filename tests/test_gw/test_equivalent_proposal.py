# -*- coding: utf-8 -*-
"""
Tests to ensure that the new GW proposal is equivalent to the old method
"""
import copy
import logging

import numpy as np

import pytest

from nessai.model import Model as BaseModel
from nessai.gw.proposal import GWFlowProposal
from nessai.gw.legacy import LegacyGWFlowProposal
from nessai.flowsampler import FlowSampler
from nessai.livepoint import dict_to_live_points

try:
    import bilby
except ImportError:
    print('Could not import bilby, some tests will not work')


logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def injection_parameters():
    injection_parameters = dict(
        mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
        phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4,
        psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375,
        dec=-1.2108)
    return injection_parameters


@pytest.fixture(scope='function')
def legacy_kwargs():
    kwargs = {
        "boundary_inversion": True,
        "update_bounds": True,
        "rescale_parameters": True,
        "analytic_priors": True,
        "reparameterisations": {
            "rescale_angles": ["psi"],
            "inversion": True,
            "convert_to_angle": {"a_1": "split", "a_2": "duplicate"},
            "use_x_prime_prior": True,
            "uniform_distance_parameter": False,
            "spin_conversion": False
        },
        "detect_edges": True,
        "inversion_type": {"mass_ratio": "duplicate", "dc3": "duplicate"},
    }
    return kwargs


@pytest.fixture(scope='function')
def kwargs(legacy_kwargs):
    kwargs = copy.deepcopy(legacy_kwargs)
    kwargs['reparameterisations'] = {
        "mass_ratio": {"reparameterisation": "mass_ratio", "prior": "uniform"},
        "chirp_mass": {"reparameterisation": "mass", "prior": "uniform"},
        "luminosity_distance": {"reparameterisation": "distance",
                                "prior": "uniform-comoving-volume"},
        "geocent_time": {"reparameterisation": "time", "prior": "uniform",
                         "update_bounds": False},
        "sky-ra-dec": {"parameters": ["ra", "dec"], "prior": "isotropic"},
        "psi": {"reparameterisation": "angle-pi"},
        "theta_jn": {"reparameterisation": "angle-sine"},
        "tilt_1": {"reparameterisation": "angle-sine"},
        "tilt_2": {"reparameterisation": "angle-sine"},
        "phi_12": {"reparameterisation": "angle-2pi"},
        "phi_jl": {"reparameterisation": "angle-2pi"},
        "a_1": {"reparameterisation": "to-cartesian", "prior": "uniform"},
        "a_2": {"reparameterisation": "to-cartesian", "prior": "uniform",
                "mode": "duplicate"}
    }
    kwargs.pop('inversion_type')
    return kwargs


@pytest.mark.requires('bilby')
@pytest.mark.requires('lal')
@pytest.mark.requires('astropy')
@pytest.mark.parametrize(
    'parameters',
    [
        ['psi', 'chirp_mass'],
        ['phi_12', 'chirp_mass'],
        ['phi_jl', 'chirp_mass'],
        ['a_1', 'chirp_mass'],
        ['a_2', 'chirp_mass'],
        ['ra', 'dec'],
        ['chirp_mass', 'geocent_time'],
        ['chirp_mass', 'mass_ratio'],
        ['chirp_mass', 'luminosity_distance'],
    ]
)
@pytest.mark.parametrize('compute_radius', [False, True])
def test_parameter(parameters, injection_parameters, kwargs, legacy_kwargs,
                   compute_radius, tmpdir):
    """
    Test that the two proposal methods are equivalent for a set of parameters.

    Checks if the resulting x prime parameters are the same and their
    log-Jacobian determinants are also the same.
    """
    outdir = str(tmpdir.mkdir('test'))
    priors = bilby.gw.prior.BBHPriorDict()
    fixed_params = ["chirp_mass", "mass_ratio", "phi_12", "phi_jl", "a_1",
                    "a_2", "tilt_1", "tilt_2", "ra", "dec",
                    "luminosity_distance", "geocent_time", "theta_jn", "psi"]
    try:
        fixed_params.remove(parameters)
    except ValueError:
        for p in parameters:
            fixed_params.remove(p)
    priors['geocent_time'] = bilby.core.prior.Uniform(
            minimum=injection_parameters['geocent_time'] - 0.1,
            maximum=injection_parameters['geocent_time'] + 0.1,
            name='geocent_time', latex_label='$t_c$', unit='$s$')
    for key in fixed_params:
        if key in injection_parameters:
            priors[key] = injection_parameters[key]
        else:
            priors.pop(key)

    priors['mass_1'] = injection_parameters['mass_1']
    priors['mass_2'] = injection_parameters['mass_2']

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=1, sampling_frequency=256,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole)

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=['H1'],
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=True,
        distance_marginalization=False,
        time_marginalization=False, reference_frame='sky'
        )

    likelihood = bilby.core.likelihood.ZeroLikelihood(likelihood)

    def log_prior(theta):
        return priors.ln_prob(theta, axis=0)

    search_parameter_keys = []
    for key in priors:
        if isinstance(priors[key], bilby.core.prior.Prior) \
                and priors[key].is_fixed is False:
            search_parameter_keys.append(key)

    def log_likelihood(theta):
        params = {
            key: t for key, t in zip(search_parameter_keys, theta)}

        likelihood.parameters.update(params)
        return likelihood.log_likelihood_ratio()

    class Model(BaseModel):
        def __init__(self, names, priors):
            self.names = names
            self.priors = priors
            self._update_bounds()

        @staticmethod
        def log_likelihood(x, **kwargs):
            theta = [x[n].item() for n in search_parameter_keys]
            return log_likelihood(theta)

        @staticmethod
        def log_prior(x, names=None, **kwargs):
            if names is None:
                names = search_parameter_keys
            theta = {n: x[n] for n in names}
            return log_prior(theta)

        def _update_bounds(self):
            self.bounds = \
                {key: [self.priors[key].minimum, self.priors[key].maximum]
                 for key in self.names}

        def new_point(self, N=1):
            prior_samples = self.priors.sample(size=N)
            samples = {n: prior_samples[n] for n in self.names}
            self._update_bounds()
            return dict_to_live_points(samples)

        def new_point_log_prob(self, x):
            return self.log_prior(x)

    reparameterisations = kwargs.pop('reparameterisations')
    for k in list(reparameterisations.keys()):
        if k == 'sky-ra-dec':
            if any(p in fixed_params for p in ['ra', 'dec']):
                del reparameterisations['sky-ra-dec']
        elif k not in search_parameter_keys:
            del reparameterisations[k]

    if 'luminosity_distance' in parameters:
        legacy_kwargs['reparameterisations']['uniform_distance_parameter'] = \
                True

    model = Model(search_parameter_keys, priors)
    sampler = FlowSampler(model, output=outdir,
                          flow_class=LegacyGWFlowProposal, **legacy_kwargs)

    new_sampler = FlowSampler(model, output=outdir,
                              reparameterisations=reparameterisations,
                              flow_class=GWFlowProposal, **kwargs)

    sampler.ns._flow_proposal.initialise()
    new_sampler.ns._flow_proposal.initialise()

    sampler.ns._uninformed_proposal.populate(N=100)
    x = sampler.ns._uninformed_proposal.samples

    orig_proposal = sampler.ns._flow_proposal
    new_proposal = new_sampler.ns._flow_proposal

    tests = ['lower', 'upper', False]
    test_results = {t: False for t in tests}
    for test in tests:
        orig_proposal.check_state(x)
        np.random.seed(1234)
        x_prime, lj = orig_proposal.rescale(
            x, test=test, compute_radius=compute_radius)
        log_p = orig_proposal.x_prime_log_prior(x_prime)

        x_re, lj_re = orig_proposal.inverse_rescale(x_prime)

        new_proposal.check_state(x)
        np.random.seed(1234)
        x_prime_new, lj_new = new_proposal.rescale(
            x, test=test, compute_radius=compute_radius)
        log_p_new = new_proposal.x_prime_log_prior(x_prime_new)

        x_re_new, lj_re_new = new_proposal.inverse_rescale(x_prime_new)

        for n in model.names:
            np.testing.assert_array_almost_equal(x_re[n], x_re_new[n])

        np.testing.assert_array_equal(lj, lj_new)
        np.testing.assert_array_equal(lj_re, lj_re_new)
        np.testing.assert_array_equal(log_p, log_p_new)
        try:
            np.testing.assert_array_equal(x_prime, x_prime_new)
        except AssertionError:
            flag = {n: False for n in x_prime.dtype.names}
            for n in x_prime.dtype.names:
                for nn in x_prime_new.dtype.names:
                    if np.allclose(x_prime[n], x_prime_new[nn]):
                        logger.critical(f'{n} is equivalent to {nn}')
                        flag[n] = True

            assert all(v for v in flag.values()), print(f'Flags: {flag}')
        test_results[test] = True

    assert all(t for t in test_results.values())

    if getattr(orig_proposal, '_rescaled_min'):
        logger.info('Checking prime prior bounds')
        test_results = {t: False for t in tests}
        for test in tests:
            logger.info(f'Testing with inversion and test={test}')
            orig_proposal.check_state(x)
            np.random.seed(1234)
            t, lj = orig_proposal.rescale(x, test=test)
            new_proposal.check_state(x)
            np.random.seed(1234)
            t_new, lj_new = new_proposal.rescale(x, test=test)
            min_vals = {
                n: False for n, r in new_proposal._reparameterisation.items()
                if r.prime_prior_bounds is not None
            }
            max_vals = {
                n: False for n, r in new_proposal._reparameterisation.items()
                if r.prime_prior_bounds is not None
            }
            for k, r in new_proposal._reparameterisation.items():
                if r.prime_prior_bounds is None:
                    continue
                for pb in r.prime_prior_bounds.values():
                    for vmin in orig_proposal._rescaled_min.values():
                        logger.warning([k, pb, vmin])
                        if np.isclose(pb[0], vmin):
                            min_vals[k] = True
                    for vmax in orig_proposal._rescaled_max.values():
                        logger.warning([k, pb, vmax])
                        if np.isclose(pb[1], vmax):
                            max_vals[k] = True

            assert all(t for t in min_vals.values()), print(min_vals)
            assert all(t for t in max_vals.values()), print(max_vals)
            test_results[test] = True

        assert all(t for t in test_results.values())
