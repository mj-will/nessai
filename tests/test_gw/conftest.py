"""Configuration for the GW tests"""
from typing import Callable

from nessai.livepoint import dict_to_live_points
from nessai.model import Model as BaseModel
import pytest


@pytest.fixture()
def injection_parameters():
    injection_parameters = dict(
        mass_ratio=0.9,
        chirp_mass=25.0,
        a_1=0.4,
        a_2=0.3,
        tilt_1=0.5,
        tilt_2=1.0,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=2000.0,
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
    )
    return injection_parameters


@pytest.fixture()
def get_bilby_gw_model() -> Callable:
    """Return a function will provide a nessai model given parameters
    and an injection.
    """

    def get_model(parameters, injection_parameters) -> BaseModel:

        import bilby

        priors = bilby.gw.prior.BBHPriorDict()
        fixed_params = [
            "chirp_mass",
            "mass_ratio",
            "phi_12",
            "phi_jl",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "ra",
            "dec",
            "luminosity_distance",
            "geocent_time",
            "theta_jn",
            "psi",
            "phase",
        ]
        try:
            fixed_params.remove(parameters)
        except ValueError:
            for p in parameters:
                fixed_params.remove(p)
        priors["geocent_time"] = bilby.core.prior.Uniform(
            minimum=injection_parameters["geocent_time"] - 0.1,
            maximum=injection_parameters["geocent_time"] + 0.1,
            name="geocent_time",
            latex_label="$t_c$",
            unit="$s$",
        )
        for key in fixed_params:
            if key in injection_parameters:
                priors[key] = injection_parameters[key]

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=1,
            sampling_frequency=256,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,  # noqa
        )

        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=["H1"],
            waveform_generator=waveform_generator,
            priors=priors,
            phase_marginalization="phase" not in fixed_params,
            distance_marginalization=False,
            time_marginalization=False,
            reference_frame="sky",
        )

        likelihood = bilby.core.likelihood.ZeroLikelihood(likelihood)

        def log_prior(theta):
            return priors.ln_prob(theta, axis=0)

        search_parameter_keys = []
        for key in priors:
            if (
                isinstance(priors[key], bilby.core.prior.Prior)
                and priors[key].is_fixed is False
            ):
                search_parameter_keys.append(key)

        def log_likelihood(theta):
            params = {key: t for key, t in zip(search_parameter_keys, theta)}

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
                self.bounds = {
                    key: [self.priors[key].minimum, self.priors[key].maximum]
                    for key in self.names
                }

            def new_point(self, N=1):
                prior_samples = self.priors.sample(size=N)
                samples = {n: prior_samples[n] for n in self.names}
                self._update_bounds()
                return dict_to_live_points(samples)

            def new_point_log_prob(self, x):
                return self.log_prior(x)

        return Model(search_parameter_keys, priors)

    return get_model
