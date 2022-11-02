# -*- coding: utf-8 -*-
"""
Tests to ensure that the new GW proposal is equivalent to the old method
"""
import copy
import logging

import numpy as np

import pytest

from nessai import config
from nessai.gw.proposal import GWFlowProposal
from nessai.gw.legacy import LegacyGWFlowProposal
from nessai.flowsampler import FlowSampler
from nessai.utils.testing import assert_structured_arrays_equal

logger = logging.getLogger(__name__)

# Dictionary to translate new parameter names to legacy names.
EQUIVALENT_PARAMETERS = dict(
    mass_ratio_prime="mass_ratio_inv",
    luminosity_distance_prime="dc3",
    geocent_time_prime="time",
    ra_dec_x="sky_x",
    ra_dec_y="sky_y",
    ra_dec_z="sky_z",
)


@pytest.fixture(autouse=True)
def update_config():
    """Configure the config to match the values for the legacy proposal"""
    original = copy.copy(config.NON_SAMPLING_DEFAULTS)
    config.NON_SAMPLING_DEFAULTS = [0.0, 0.0, 0]
    yield
    config.NON_SAMPLING_DEFAULTS = original


@pytest.fixture(scope="function")
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
            "spin_conversion": False,
        },
        "detect_edges": True,
        "inversion_type": {"mass_ratio": "duplicate", "dc3": "duplicate"},
    }
    return kwargs


@pytest.fixture(scope="function")
def kwargs(legacy_kwargs):
    kwargs = copy.deepcopy(legacy_kwargs)
    kwargs["reparameterisations"] = {
        "mass_ratio": {"reparameterisation": "mass_ratio", "prior": "uniform"},
        "chirp_mass": {"reparameterisation": "mass", "prior": "uniform"},
        "luminosity_distance": {
            "reparameterisation": "distance",
            "prior": "uniform-comoving-volume",
        },
        "geocent_time": {
            "reparameterisation": "time",
            "prior": "uniform",
            "update_bounds": False,
        },
        "sky-ra-dec": {"parameters": ["ra", "dec"], "prior": "isotropic"},
        "psi": {"reparameterisation": "angle-pi"},
        "theta_jn": {"reparameterisation": "angle-sine"},
        "tilt_1": {"reparameterisation": "angle-sine"},
        "tilt_2": {"reparameterisation": "angle-sine"},
        "phi_12": {"reparameterisation": "angle-2pi"},
        "phi_jl": {"reparameterisation": "angle-2pi"},
        "a_1": {"reparameterisation": "to-cartesian", "prior": "uniform"},
        "a_2": {
            "reparameterisation": "to-cartesian",
            "prior": "uniform",
            "mode": "duplicate",
        },
    }
    kwargs.pop("inversion_type")
    return kwargs


@pytest.mark.requires("bilby")
@pytest.mark.requires("lal")
@pytest.mark.requires("astropy")
@pytest.mark.parametrize(
    "parameters",
    [
        ["psi", "chirp_mass"],
        ["phi_12", "chirp_mass"],
        ["phi_jl", "chirp_mass"],
        ["a_1", "chirp_mass"],
        ["a_2", "chirp_mass"],
        ["ra", "dec"],
        ["chirp_mass", "geocent_time"],
        ["chirp_mass", "mass_ratio"],
        ["chirp_mass", "luminosity_distance"],
        ["mass_ratio", "chirp_mass"],
        ["mass_ratio", "ra", "dec"],
        ["ra", "dec", "mass_ratio"],
    ],
)
@pytest.mark.parametrize("compute_radius", [False, True])
@pytest.mark.integration_test
def test_parameter(
    get_bilby_gw_model,
    parameters,
    injection_parameters,
    kwargs,
    legacy_kwargs,
    compute_radius,
    tmpdir,
):
    """
    Test that the two proposal methods are equivalent for a set of parameters.

    Checks if the resulting x prime parameters are the same and their
    log-Jacobian determinants are also the same.
    """
    outdir = str(tmpdir.mkdir("test"))

    model = get_bilby_gw_model(parameters, injection_parameters)

    reparameterisations = kwargs.pop("reparameterisations")
    for k in list(reparameterisations.keys()):
        if k == "sky-ra-dec":
            if not all(p in model.names for p in ["ra", "dec"]):
                del reparameterisations["sky-ra-dec"]
        elif k not in model.names:
            del reparameterisations[k]

    if "luminosity_distance" in parameters:
        legacy_kwargs["reparameterisations"][
            "uniform_distance_parameter"
        ] = True

    sampler = FlowSampler(
        model, output=outdir, flow_class=LegacyGWFlowProposal, **legacy_kwargs
    )

    new_sampler = FlowSampler(
        model,
        output=outdir,
        reparameterisations=reparameterisations,
        flow_class=GWFlowProposal,
        **kwargs,
    )

    sampler.ns._flow_proposal.initialise()
    new_sampler.ns._flow_proposal.initialise()

    sampler.ns._uninformed_proposal.populate(N=100)
    x = sampler.ns._uninformed_proposal.samples

    orig_proposal = sampler.ns._flow_proposal
    new_proposal = new_sampler.ns._flow_proposal

    tests = ["lower", "upper", False]
    test_results = {t: False for t in tests}
    for test in tests:
        orig_proposal.check_state(x)
        np.random.seed(1234)
        x_prime, lj = orig_proposal.rescale(
            x, test=test, compute_radius=compute_radius
        )
        log_p = orig_proposal.x_prime_log_prior(x_prime)

        x_re, lj_re = orig_proposal.inverse_rescale(x_prime)

        new_proposal.check_state(x)
        np.random.seed(1234)
        x_prime_new, lj_new = new_proposal.rescale(
            x, test=test, compute_radius=compute_radius
        )
        log_p_new = new_proposal.x_prime_log_prior(x_prime_new)

        x_re_new, lj_re_new = new_proposal.inverse_rescale(x_prime_new)

        for n in model.names:
            np.testing.assert_array_almost_equal(x_re[n], x_re_new[n])

        np.testing.assert_array_almost_equal_nulp(lj, lj_new)
        np.testing.assert_array_almost_equal_nulp(lj_re, lj_re_new)
        np.testing.assert_array_equal(log_p, log_p_new)
        try:
            assert_structured_arrays_equal(x_prime, x_prime_new)
        except (AssertionError, TypeError):
            flag = {n: False for n in x_prime_new.dtype.names}
            for n in x_prime_new.dtype.names:
                # Get equivalent name
                # If not included, assume it should be the same.
                legacy_name = EQUIVALENT_PARAMETERS.get(n, n)
                if np.allclose(x_prime[legacy_name], x_prime_new[n]):
                    logger.critical(f"{n} is equivalent to {legacy_name}")
                    flag[n] = True
            assert all(v for v in flag.values()), print(f"Flags: {flag}")
        test_results[test] = True

    assert all(t for t in test_results.values())

    if getattr(orig_proposal, "_rescaled_min"):
        logger.info("Checking prime prior bounds")
        test_results = {t: False for t in tests}
        for test in tests:
            logger.info(f"Testing with inversion and test={test}")
            orig_proposal.check_state(x)
            np.random.seed(1234)
            t, lj = orig_proposal.rescale(x, test=test)
            new_proposal.check_state(x)
            np.random.seed(1234)
            t_new, lj_new = new_proposal.rescale(x, test=test)
            min_vals = {
                n: False
                for n, r in new_proposal._reparameterisation.items()
                if r.prime_prior_bounds is not None
            }
            max_vals = {
                n: False
                for n, r in new_proposal._reparameterisation.items()
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
