#!/usr/bin/env python
"""
Example of running nessai with calibration uncertainty. Nessai can run with
calibration uncertainty but it has not been extensively tested.

This example should take ~30 minutes to run.

Adapted from the example included in bilby.
"""
import numpy as np
import bilby

# Standard configuration using bilby
duration = 4.0
sampling_frequency = 2048.0

outdir = "./outdir/"
label = "calibration_example"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(150914)

# Injection parameters for a GW150914-like BBH.
injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
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

waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2", reference_frequency=50.0
)

# Create the waveform_generator
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameters=injection_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
for ifo in ifos:
    injection_parameters.update(
        {
            "recalib_{}_amplitude_{}".format(ifo.name, ii): 0.05
            for ii in range(5)
        }
    )
    injection_parameters.update(
        {"recalib_{}_phase_{}".format(ifo.name, ii): 0.01 for ii in range(5)}
    )
    ifo.calibration_model = bilby.gw.calibration.CubicSpline(
        prefix="recalib_{}_".format(ifo.name),
        minimum_frequency=ifo.minimum_frequency,
        maximum_frequency=ifo.maximum_frequency,
        n_points=5,
    )
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration
)
ifos.inject_signal(
    parameters=injection_parameters, waveform_generator=waveform_generator
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
fixed_parameters = [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_jl",
    "phi_12",
    "psi",
    "geocent_time",
    "luminosity_distance",
]

for key in injection_parameters:
    if "recalib" in key:
        priors[key] = injection_parameters[key]
    if key in fixed_parameters:
        priors[key] = injection_parameters[key]

calib_amplitudes = [
    "recalib_H1_amplitude_0",
    "recalib_L1_amplitude_0",
    "recalib_H1_amplitude_1",
    "recalib_L1_amplitude_1",
]

calib_phases = [
    "recalib_H1_phase_0",
    "recalib_L1_phase_0",
    "recalib_H1_phase_1",
    "recalib_L1_phase_1",
]

calib_parameters = calib_amplitudes + calib_phases

# Entry 8 is the detector (H or L)
for name in calib_amplitudes:
    priors[name] = bilby.prior.Gaussian(
        mu=0, sigma=0.2, name=name, latex_label=f"{name[8]}1 $A_{name[-1]}$"
    )

for name in calib_phases:
    priors[name] = bilby.prior.Gaussian(
        mu=0, sigma=0.1, name=name, latex_label=f"{name[8]}1 $P_{name[-1]}$"
    )

# Initialise the likelihood by passing in the interferometer data (IFOs) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
    phase_marginalization=True,
)

# Run sampler.
# By default the calibration parameters will not be reparameterised.
# Here we show how you could set a reparameterisation for these parameters.
result = bilby.run_sampler(
    resume=False,
    likelihood=likelihood,
    priors=priors,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    plot=True,
    sampler="nessai",
    flow_class="gwflowproposal",
    reparameterisations={"null": {"parameters": calib_parameters}},
)
