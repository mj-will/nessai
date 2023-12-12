# -*- coding: utf-8 -*-
"""
Integration tests for running the sampler with different configurations.
"""
import logging
import os
from scipy.stats import norm
import torch
import pytest
import numpy as np
from unittest.mock import patch

from nessai.flowsampler import FlowSampler
from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model
from nessai.utils.testing import IntegrationTestModel


torch.set_num_threads(1)


@pytest.mark.slow_integration_test
def test_sampling_with_rescale(integration_model, flow_config, tmpdir):
    """
    Test sampling with rescaling. Checks that flow is trained.
    """
    output = str(tmpdir.mkdir("w_rescale"))
    fp = FlowSampler(
        integration_model,
        output=output,
        resume=False,
        nlive=100,
        plot=False,
        flow_config=flow_config,
        training_frequency=10,
        maximum_uninformed=9,
        rescale_parameters=True,
        seed=1234,
        max_iteration=11,
        poolsize=10,
    )
    fp.run()
    assert fp.ns.proposal.flow.weights_file is not None
    assert fp.ns.proposal.training_count == 1


@pytest.mark.slow_integration_test
def test_sampling_with_inversion(integration_model, flow_config, tmpdir):
    """
    Test sampling with inversion. Checks that flow is trained.
    """
    output = str(tmpdir.mkdir("w_rescale"))
    fp = FlowSampler(
        integration_model,
        output=output,
        resume=False,
        nlive=100,
        plot=False,
        flow_config=flow_config,
        training_frequency=10,
        maximum_uninformed=9,
        rescale_parameters=True,
        seed=1234,
        max_iteration=11,
        poolsize=10,
        boundary_inversion=True,
        update_bounds=True,
    )
    fp.run()
    reparams = list(fp.ns.proposal._reparameterisation.values())
    assert len(reparams) == 1
    assert reparams[0].parameters == ["x_0", "x_1"]
    assert list(reparams[0].boundary_inversion.keys()) == ["x_0", "x_1"]
    assert fp.ns.proposal.flow.weights_file is not None
    assert fp.ns.proposal.training_count == 1


def test_sampling_regex_reparams(model, flow_config, tmp_path):
    """Test using regex to specify reparameterisations"""
    model._names = ["x_0", "x_1"]
    model._bounds = {"x_0": [-5, 5], "x_1": [-5, 5]}

    fs = FlowSampler(
        model,
        nlive=100,
        output=tmp_path / "test_regex",
        flow_config=flow_config,
        reparameterisations={"z-score": {"parameters": ["x.*"]}},
        maximum_uninformed=50,
        max_iteration=100,
        plot=False,
    )
    fs.run()
    assert fs.ns._flow_proposal.rescaled_names == ["x_0_prime", "x_1_prime"]


@pytest.mark.slow_integration_test
def test_sampling_without_rescale(integration_model, flow_config, tmpdir):
    """
    Test sampling without rescaling. Checks that flow is trained.
    """
    output = str(tmpdir.mkdir("wo_rescale"))
    fp = FlowSampler(
        integration_model,
        output=output,
        resume=False,
        nlive=100,
        plot=False,
        flow_config=flow_config,
        training_frequency=10,
        maximum_uninformed=9,
        rescale_parameters=False,
        seed=1234,
        max_iteration=11,
        poolsize=10,
    )
    fp.run()
    assert fp.ns.proposal.flow.weights_file is not None
    assert fp.ns.proposal.training_count == 1


@pytest.mark.slow_integration_test
def test_sampling_with_maf(integration_model, flow_config, tmpdir):
    """
    Test sampling with MAF. Checks that flow is trained but does not
    check convergence.
    """
    flow_config["model_config"]["ftype"] = "maf"
    output = str(tmpdir.mkdir("maf"))
    fp = FlowSampler(
        integration_model,
        output=output,
        resume=False,
        nlive=100,
        plot=False,
        flow_config=flow_config,
        training_frequency=10,
        maximum_uninformed=9,
        rescale_parameters=True,
        seed=1234,
        max_iteration=11,
        poolsize=10,
    )
    fp.run()
    assert fp.ns.proposal.flow.weights_file is not None
    assert fp.ns.proposal.training_count == 1


@pytest.mark.slow_integration_test
@pytest.mark.parametrize("analytic", [False, True])
def test_sampling_uninformed(integration_model, flow_config, tmpdir, analytic):
    """
    Test running the sampler with the two uninformed proposal methods.
    """
    output = str(tmpdir.mkdir("uninformed"))
    fp = FlowSampler(
        integration_model,
        output=output,
        resume=False,
        nlive=100,
        plot=False,
        flow_config=flow_config,
        training_frequency=None,
        maximum_uninformed=10,
        rescale_parameters=True,
        seed=1234,
        max_iteration=11,
        poolsize=10,
        analytic_priors=analytic,
    )
    fp.run()


@pytest.mark.slow_integration_test
@pytest.mark.parametrize("parallelise_prior", [True, False])
def test_sampling_with_n_pool(
    integration_model,
    flow_config,
    tmpdir,
    mp_context,
    parallelise_prior,
):
    """
    Test running the sampler with multiprocessing.
    """
    output = str(tmpdir.mkdir("pool"))
    with patch("multiprocessing.Pool", mp_context.Pool), patch(
        "nessai.utils.multiprocessing.multiprocessing.get_start_method",
        mp_context.get_start_method,
    ):
        fp = FlowSampler(
            integration_model,
            output=output,
            resume=False,
            nlive=100,
            plot=False,
            flow_config=flow_config,
            training_frequency=10,
            maximum_uninformed=9,
            rescale_parameters=True,
            seed=1234,
            max_iteration=11,
            poolsize=10,
            pytorch_threads=2,
            n_pool=2,
            parallelise_prior=parallelise_prior,
        )
    fp.run()
    assert fp.ns.proposal.flow.weights_file is not None
    assert fp.ns.proposal.training_count == 1
    assert os.path.exists(os.path.join(output, "result.hdf5"))


@pytest.mark.slow_integration_test
def test_sampling_resume(model, flow_config, tmpdir):
    """
    Test resuming the sampler.
    """
    output = str(tmpdir.mkdir("resume"))
    fp = FlowSampler(
        model,
        output=output,
        resume=True,
        nlive=100,
        plot=False,
        flow_config=flow_config,
        training_frequency=10,
        maximum_uninformed=9,
        rescale_parameters=True,
        checkpoint_on_iteration=True,
        checkpoint_interval=5,
        seed=1234,
        max_iteration=11,
        poolsize=10,
    )
    fp.run()
    assert os.path.exists(os.path.join(output, "nested_sampler_resume.pkl"))

    fp = FlowSampler(
        model,
        output=output,
        resume=True,
        flow_config=flow_config,
    )
    assert fp.ns.iteration == 11
    fp.ns.max_iteration = 21
    fp.run()
    assert fp.ns.iteration == 21
    assert os.path.exists(
        os.path.join(output, "nested_sampler_resume.pkl.old")
    )


@pytest.mark.slow_integration_test
def test_sampling_resume_w_pool(
    integration_model, flow_config, tmpdir, mp_context
):
    """
    Test resuming the sampler with a pool.
    """
    output = str(tmpdir.mkdir("resume"))
    with patch("multiprocessing.Pool", mp_context.Pool), patch(
        "nessai.utils.multiprocessing.multiprocessing.get_start_method",
        mp_context.get_start_method,
    ):
        fp = FlowSampler(
            integration_model,
            output=output,
            resume=True,
            nlive=100,
            plot=False,
            flow_config=flow_config,
            training_frequency=10,
            maximum_uninformed=9,
            rescale_parameters=True,
            checkpoint_on_iteration=True,
            checkpoint_interval=5,
            seed=1234,
            max_iteration=11,
            poolsize=10,
            n_pool=1,
        )
    assert fp.ns.model.n_pool == 1
    fp.run()
    assert os.path.exists(os.path.join(output, "nested_sampler_resume.pkl"))
    # Make sure the pool is already closed
    integration_model.close_pool()

    with patch("multiprocessing.Pool", mp_context.Pool), patch(
        "nessai.utils.multiprocessing.multiprocessing.get_start_method",
        mp_context.get_start_method,
    ):
        fp = FlowSampler(
            integration_model,
            output=output,
            resume=True,
            flow_config=flow_config,
            n_pool=1,
        )
    assert fp.ns.model.n_pool == 1
    assert fp.ns.iteration == 11
    fp.ns.max_iteration = 21
    fp.run()
    assert fp.ns.iteration == 21
    assert os.path.exists(
        os.path.join(output, "nested_sampler_resume.pkl.old")
    )


@pytest.mark.slow_integration_test
def test_sampling_resume_no_max_uninformed(
    integration_model, flow_config, tmpdir
):
    """
    Test resuming the sampler when there is no maximum iteration for
    the uinformed sampling.

    This test makes sure the correct proposal is loaded after resuming
    and re-initialising the sampler.
    """
    output = str(tmpdir.mkdir("resume"))
    fp = FlowSampler(
        integration_model,
        output=output,
        resume=True,
        nlive=100,
        plot=False,
        flow_config=flow_config,
        training_frequency=10,
        maximum_uninformed=9,
        rescale_parameters=True,
        seed=1234,
        max_iteration=11,
        checkpoint_on_iteration=True,
        checkpoint_interval=5,
        poolsize=10,
    )
    fp.run()
    assert os.path.exists(os.path.join(output, "nested_sampler_resume.pkl"))

    fp = FlowSampler(
        integration_model, output=output, resume=True, flow_config=flow_config
    )
    assert fp.ns.iteration == 11
    fp.ns.maximum_uninformed = np.inf
    fp.ns.initialise()
    assert fp.ns.proposal is fp.ns._flow_proposal
    fp.ns.max_iteration = 21
    fp.run()
    assert fp.ns.iteration == 21
    assert os.path.exists(
        os.path.join(output, "nested_sampler_resume.pkl.old")
    )


@pytest.mark.slow_integration_test
def test_resume_fallback_reparameterisation(tmpdir, model, flow_config):
    """
    Test resuming the sampler with the default reparameterisations disabled
    and the fallback set.

    This should load the same class after resuming.
    """
    from nessai.reparameterisations.rescale import ScaleAndShift

    output = str(tmpdir.mkdir("resume"))
    fp = FlowSampler(
        model,
        output=output,
        resume=True,
        nlive=100,
        plot=False,
        flow_config=flow_config,
        training_frequency=10,
        maximum_uninformed=9,
        rescale_parameters=False,
        use_default_reparameterisations=False,
        fallback_reparameterisation="z-score",
        checkpoint_on_iteration=True,
        checkpoint_interval=5,
        seed=1234,
        max_iteration=11,
        poolsize=10,
    )
    fp.run()

    reparam = fp.ns._flow_proposal._reparameterisation
    reparam = next(iter(fp.ns._flow_proposal._reparameterisation.values()))
    assert isinstance(reparam, ScaleAndShift)
    assert os.path.exists(os.path.join(output, "nested_sampler_resume.pkl"))

    fp = FlowSampler(
        model,
        output=output,
        resume=True,
        flow_config=flow_config,
    )
    assert fp.ns.iteration == 11
    fp.ns.max_iteration = 21
    fp.run()
    reparam = next(iter(fp.ns._flow_proposal._reparameterisation.values()))
    assert isinstance(reparam, ScaleAndShift)
    assert fp.ns.iteration == 21
    assert os.path.exists(
        os.path.join(output, "nested_sampler_resume.pkl.old")
    )


@pytest.mark.slow_integration_test
def test_sampling_with_infinite_prior_bounds(tmpdir):
    """
    Make sure the sampler runs when sampling a parameter with infinite prior \
        bounds.
    """
    output = str(tmpdir.mkdir("infinite_bounds"))

    class TestModel(Model):

        names = ["x", "y"]
        bounds = {"x": [0, 1], "y": [-np.inf, np.inf]}
        reparameterisations = {"x": "default", "y": None}

        def new_point(self, N=1):
            x = np.concatenate(
                [np.random.rand(N, 1), np.random.randn(N, 1)], axis=1
            )
            return numpy_array_to_live_points(x, self.names)

        def log_prior(self, x):
            log_p = np.log(self.in_bounds(x))
            log_p += norm.logpdf(x["y"])
            return log_p

        def log_likelihood(self, x):
            log_l = np.zeros(x.size)
            for n in self.names:
                log_l += norm.logpdf(x[n])
            return log_l

    fs = FlowSampler(
        TestModel(), output=output, nlive=500, plot=False, proposal_plots=False
    )
    fs.run(plot=False)
    assert fs.ns.condition <= 0.1


@pytest.mark.slow_integration_test
def test_constant_volume_mode(integration_model, tmpdir):
    """Test sampling in constant volume mode"""
    output = str(tmpdir.mkdir("test"))
    fs = FlowSampler(
        integration_model,
        output=output,
        nlive=500,
        plot=False,
        proposal_plots=False,
        constant_volume_mode=True,
    )
    fs.run(plot=False)


@pytest.mark.slow_integration_test
def test_sampling_with_plotting(integration_model, tmpdir):
    """Test sampling with plots enabled"""
    output = str(tmpdir.mkdir("test"))
    fs = FlowSampler(
        integration_model,
        output=output,
        nlive=100,
        plot=True,
        proposal_plots=True,
    )
    fs.run(plot=True)
    assert os.path.exists(os.path.join(output, "proposal", "pool_0.png"))
    assert os.path.exists(os.path.join(output, "proposal", "pool_0_log_q.png"))


@pytest.mark.slow_integration_test
def test_truncate_log_q(integration_model, tmpdir):
    """Test sampling with truncate_log_q"""
    output = str(tmpdir.mkdir("test"))
    fs = FlowSampler(
        integration_model,
        output=output,
        nlive=500,
        plot=False,
        proposal_plots=False,
        constant_volume_mode=False,
        latent_prior="flow",
        truncate_log_q=True,
    )
    fs.run(plot=False)
    assert fs.ns.finalised


@pytest.mark.slow_integration_test
def test_prior_sampling(integration_model, tmpdir):
    """Test prior sampling"""
    output = str(tmpdir.mkdir("prior_sampling"))
    fs = FlowSampler(
        integration_model,
        output=output,
        nlive=100,
        plot=False,
        prior_sampling=True,
    )
    fs.run(plot=False)

    assert len(fs.nested_samples) == 100
    assert np.isfinite(fs.logZ)


@pytest.mark.slow_integration_test
def test_sampling_resume_finalised(integration_model, tmp_path):
    """Test resuming the sampler after it finishes sampling"""
    output = tmp_path / "output"
    output.mkdir()
    fs = FlowSampler(
        integration_model,
        output=output,
        nlive=100,
        plot=False,
    )
    fs.run(save=False, plot=False)
    assert os.path.exists(fs.ns.resume_file)

    fs = FlowSampler(
        integration_model,
        output=output,
        nlive=100,
        plot=False,
    )

    fs.run(save=False, plot=False)


@pytest.mark.slow_integration_test
def test_debug_log_level(integration_model, tmpdir):
    """Test running with debug log-level."""
    logger = logging.getLogger("nessai")
    original_level = logger.level
    logger.setLevel("DEBUG")
    output = str(tmpdir.mkdir("debug_logging"))
    fs = FlowSampler(
        integration_model,
        output=output,
        nlive=100,
        plot=False,
    )
    fs.run(plot=False)
    logger.setLevel(original_level)


@pytest.mark.slow_integration_test
def test_disable_vectorisation(tmp_path):
    """Assert vectorisation can be disabled"""

    class TestModel(IntegrationTestModel):
        def log_likelihood(self, x):
            # AssertionError won't be caught by nessai
            assert not (x.size > 1)
            return super().log_likelihood(x)

    output = tmp_path / "disable_vec"
    output.mkdir()

    model = TestModel()

    fs = FlowSampler(
        model,
        output=output,
        nlive=100,
        disable_vectorisation=True,
        plot=False,
    )
    fs.run(plot=False)


@pytest.mark.slow_integration_test
def test_likelihood_chunksize(tmp_path):
    """Assert likelihood chunksize limits number of samples in each call"""

    class TestModel(IntegrationTestModel):
        def log_likelihood(self, x):
            # AssertionError won't be caught by nessai
            assert not (x.size > self.likelihood_chunksize)
            return super().log_likelihood(x)

    output = tmp_path / "disable_vec"
    output.mkdir()

    model = TestModel()

    fs = FlowSampler(
        model, output=output, nlive=100, plot=False, likelihood_chunksize=10
    )
    fs.run(plot=False, save=False)


@pytest.mark.slow_integration_test
def test_allow_multi_valued_likelihood(tmp_path):
    """Assert a multi valued likelihood can be sampled from"""

    class TestModel(IntegrationTestModel):
        def log_likelihood(self, x):
            return super().log_likelihood(x) + 1e-10 * np.random.randn(x.size)

    output = tmp_path / "multi_value"
    output.mkdir()

    model = TestModel()

    fs = FlowSampler(
        model,
        output=output,
        nlive=100,
        plot=False,
        allow_multi_valued_likelihood=True,
    )
    fs.run(plot=False, save=False)


@pytest.mark.integration_test
def test_invalid_keyword_argument(integration_model, tmp_path):
    """Assert an error is raised if a keyword argument is unknown"""

    output = tmp_path / "kwargs_error"
    output.mkdir()

    with pytest.raises(
        RuntimeError,
        match="Unknown kwargs for FlowProposal: {'not_a_valid_kwarg'}.",
    ):
        FlowSampler(
            integration_model,
            output=output,
            nlive=100,
            plot=False,
            not_a_valid_kwarg=True,
        )


@pytest.mark.parametrize("extension", ["hdf5", "h5", "json"])
@pytest.mark.slow_integration_test
def test_sampling_result_extension(integration_model, tmp_path, extension):
    """Assert the correct extension is used"""
    output = tmp_path / "test"
    output.mkdir()
    fs = FlowSampler(
        integration_model,
        output=output,
        nlive=100,
        plot=False,
        proposal_plots=False,
        result_extension=extension,
    )
    fs.run(plot=False)
    assert os.path.exists(os.path.join(output, f"result.{extension}"))


@pytest.mark.slow_integration_test
def test_sampling_with_checkpoint_callback(integration_model, tmp_path):
    """Test the usage if the checkpoint callbacks"""
    import pickle

    output = tmp_path / "test_callbacks"
    output.mkdir()

    checkpoint_file = output / "test.pkl"

    def callback(state):
        with open(checkpoint_file, "wb") as f:
            pickle.dump(state, f)

    fs = FlowSampler(
        integration_model,
        output=output,
        nlive=100,
        plot=False,
        proposal_plots=False,
        checkpoint_callback=callback,
        max_iteration=100,
        checkpoint_on_iteration=True,
        checkpoint_interval=50,
    )
    fs.run(plot=False)
    assert fs.ns.iteration == 100

    del fs

    with open(checkpoint_file, "rb") as f:
        resume_data = pickle.load(f)

    fs = FlowSampler(
        integration_model,
        output=output,
        nlive=100,
        plot=False,
        proposal_plots=False,
        checkpoint_callback=callback,
        checkpoint_on_iteration=True,
        checkpoint_interval=50,
        resume_data=resume_data,
        resume=True,
    )
    fs.ns.max_iteration = 200
    fs.run()
    assert fs.ns.iteration == 200
