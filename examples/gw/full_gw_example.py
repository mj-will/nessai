#!/usr/bin/env python

"""
Example of running nessai with bilby on a gravitational wave likelihood. This
examples includes all 15 parameters for CBC and should take around 2 hours to
run.

Based on the Bilby example: https://git.ligo.org/lscsoft/bilby
"""
import bilby
import numpy as np

outdir = "./outdir/"
label = "full_gw_example"

bilby.core.utils.setup_logger(outdir=outdir, label=label)

duration = 4.0
sampling_frequency = 2048.0

np.random.seed(151226)

# Use an injection that is similar to GW150914
injection_parameters = dict(
    total_mass=66.0,
    mass_ratio=0.9,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=2000,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2", reference_frequency=50.0
)

# Create the waveform_generator
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=sampling_frequency,
    duration=duration,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=(
        bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
    ),
    waveform_arguments=waveform_arguments,
)

# Set up interferometers
ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 3,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Set up prior
priors = bilby.gw.prior.BBHPriorDict()
priors["geocent_time"] = bilby.core.prior.Uniform(
    minimum=injection_parameters["geocent_time"] - 0.1,
    maximum=injection_parameters["geocent_time"] + 0.1,
    name="geocent_time",
    latex_label="$t_c$",
    unit="$s$",
)

# Initialise the likelihood
# nessai supports the marginalisation included in bilby
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    phase_marginalization=True,
    distance_marginalization=True,
)


# Run sampler

# The `flow_class` should be set to `GWFlowProposal` for GW PE. This includes
# specific default reparameterisations for certain parameters. For example,
# it knows that theta_jn is angle with a sine prior.

result = bilby.core.sampler.run_sampler(
    likelihood=likelihood,
    priors=priors,
    outdir=outdir,
    injection_parameters=injection_parameters,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    flow_class="GWFlowProposal",
    sampler="nessai",
    resume=False,
    plot=True,
    nlive=2000,
    maximum_uninformed=4000,
    seed=150914,
    analytic_priors=True,  # Bilby priors can be sampled from directly
    flow_config=dict(model_config=dict(n_blocks=6)),
    n_pool=2,
)

# Produce corner plots
result.plot_corner()
