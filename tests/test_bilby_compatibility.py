# -*- coding: utf-8 -*-
"""Test compatibility with bilby"""
import numpy as np
import pytest


@pytest.mark.requires("bilby")
@pytest.mark.bilby_compatibility
@pytest.mark.slow_integration_test
def test_bilby_compatiblity(tmp_path):
    """Test compatibility with bilby"""
    import bilby

    outdir = tmp_path / "bilby_test"

    class GaussianLikelihood(bilby.Likelihood):
        def __init__(self):
            super().__init__(parameters={"x": None, "y": None})

        def log_likelihood(self):
            """Log-likelihood."""
            return -0.5 * (
                self.parameters["x"] ** 2.0 + self.parameters["y"] ** 2.0
            ) - np.log(2.0 * np.pi)

    priors = dict(
        x=bilby.core.prior.Uniform(-5, 5, "x"),
        y=bilby.core.prior.Uniform(-5, 5, "y"),
    )

    bilby.run_sampler(
        outdir=outdir,
        resume=False,
        plot=False,
        likelihood=GaussianLikelihood(),
        priors=priors,
        sampler="nessai",
        injection_parameters={"x": 0.0, "y": 0.0},
        analytic_priors=True,
        nlive=500,
        seed=1234,
        logging_interval=10,
        log_on_iteration=False,
    )
