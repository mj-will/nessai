# -*- coding: utf-8 -*-
"""End-to-end test of the populate method"""
import numpy as np
import pytest
import torch

from nessai.proposal import FlowProposal
from nessai.livepoint import numpy_array_to_live_points
from nessai.utils.testing import assert_structured_arrays_equal

torch.set_num_threads(1)


@pytest.mark.parametrize()
def flow_config():
    return dict(
        model_config=dict(n_neurons=1, n_blocks=1, n_layers=1, kwargs={})
    )


@pytest.mark.parametrize("expansion_fraction", [0.0, 1.0, None])
@pytest.mark.parametrize("check_acceptance", [False, True])
@pytest.mark.parametrize("rescale_parameters", [False, True])
@pytest.mark.integration_test
@pytest.mark.timeout(30)
def test_flowproposal_populate(
    tmp_path,
    model,
    flow_config,
    expansion_fraction,
    check_acceptance,
    rescale_parameters,
):
    """
    Test the populate method in the FlowProposal class with a range of
    parameters
    """
    output = tmp_path / "flowproposal"
    output.mkdir()
    n_draw = 10
    fp = FlowProposal(
        model,
        output=output,
        flow_config=flow_config,
        plot=False,
        poolsize=10,
        latent_prior="truncated_gaussian",
        expansion_fraction=expansion_fraction,
        check_acceptance=check_acceptance,
        rescale_parameters=rescale_parameters,
        max_radius=1.0,
        constant_volume_mode=False,
    )

    fp.initialise()
    worst = numpy_array_to_live_points(0.01 * np.ones(fp.dims), fp.names)
    fp.populate(worst, N=n_draw)

    assert fp.x.size == n_draw


@pytest.mark.parametrize(
    "latent_prior",
    [
        "gaussian",
        "truncated_gaussian",
        "uniform_nball",
        "uniform_nsphere",
        "uniform",
    ],
)
@pytest.mark.integration_test
@pytest.mark.timeout(30)
def test_flowproposal_populate_edge_cases(
    tmp_path, model, flow_config, latent_prior
):
    """Tests some less common settings for flowproposal"""
    output = tmp_path / "flowproposal"
    output.mkdir()
    n_draw = 2
    fp = FlowProposal(
        model,
        output=output,
        flow_config=flow_config,
        plot=False,
        poolsize=10,
        latent_prior=latent_prior,
        expansion_fraction=None,
        rescale_parameters=False,
        max_radius=0.1,
        constant_volume_mode=False,
    )

    fp.initialise()
    assert fp.rescale_parameters is False
    assert fp.parameters_to_rescale == []
    worst = numpy_array_to_live_points(0.01 * np.ones(fp.dims), fp.names)
    fp.populate(worst, N=n_draw)

    assert fp.x.size == n_draw


@pytest.mark.parametrize("plot", [False, True])
@pytest.mark.integration_test
def test_training(tmpdir, model, plot):
    """Integration test to test training the flow with and without plotting."""
    output = str(tmpdir.mkdir("test_train"))
    config = dict(max_epochs=10)
    fp = FlowProposal(
        model, output=output, plot="min", poolsize=100, flow_config=config
    )

    fp.initialise()

    x = model.new_point(500)
    fp.train(x, plot=plot)

    assert fp.training_count == 1
    assert fp.populated is False


@pytest.mark.parametrize("check_acceptance", [False, True])
@pytest.mark.parametrize("rescale_parameters", [False, True])
@pytest.mark.integration_test
@pytest.mark.timeout(30)
def test_constant_volume_mode(
    tmpdir, model, flow_config, check_acceptance, rescale_parameters
):
    """Integration test for constant volume mode.

    With q=0.8647 should get a radius of ~2.
    """
    output = str(tmpdir.mkdir("flowproposal"))
    expected_radius = 2.0
    fp = FlowProposal(
        model,
        output=output,
        flow_config=flow_config,
        plot=False,
        poolsize=10,
        constant_volume_mode=True,
        volume_fraction=0.8647,
        check_acceptance=check_acceptance,
        rescale_parameters=rescale_parameters,
    )
    fp.initialise()
    worst = numpy_array_to_live_points(0.5 * np.ones(fp.dims), fp.names)
    fp.populate(worst, N=10)
    assert fp.x.size == 10

    np.testing.assert_approx_equal(fp.r, expected_radius, 4)
    np.testing.assert_approx_equal(fp.fixed_radius, expected_radius, 4)


@pytest.mark.parametrize("boundary_inversion", [True, False])
@pytest.mark.parametrize("inversion_type", ["split", "duplicate"])
@pytest.mark.integration_test
def test_verify_rescaling_integration(
    tmp_path, model, boundary_inversion, inversion_type
):
    """Assert verify rescaling passes."""
    output = tmp_path / "test"
    output.mkdir()

    fp = FlowProposal(
        model,
        output=output,
        poolsize=10,
        boundary_inversion=boundary_inversion,
        inversion_type=inversion_type,
    )
    fp.set_rescaling()
    fp.verify_rescaling()


@pytest.mark.integration_test
def test_rescaling_integration_with_rescale_parameters(tmp_path, model):
    """Assert rescale parameters is configured correctly"""
    output = tmp_path / "test"
    output.mkdir()

    fp = FlowProposal(
        model,
        output=output,
        poolsize=10,
        rescale_parameters=True,
        rescale_bounds=[0, 1],
    )
    fp.set_rescaling()

    n = 10
    x = model.new_point(n)

    x_prime, log_j = fp.rescale(x)

    x_recon, log_j_inv = fp.inverse_rescale(x_prime)

    x_prime_expected = x_prime.copy()
    for name in model.names:
        x_prime_expected[name + "_prime"] = (
            x[name] - model.bounds[name][0]
        ) / (model.bounds[name][1] - model.bounds[name][0])
    expected_log_j = -np.log(np.prod(model.upper_bounds - model.lower_bounds))

    assert len(log_j) == n
    assert (log_j == expected_log_j).all()
    np.testing.assert_array_equal(log_j, -log_j_inv)

    assert_structured_arrays_equal(x_prime, x_prime_expected)
    assert_structured_arrays_equal(x_recon, x)


@pytest.mark.integration_test
def test_rescaling_integration_no_rescaling(tmp_path, model):
    """Assert setting rescale_parameters=False does not rescale any
    parameters.
    """
    output = tmp_path / "test"
    output.mkdir()

    fp = FlowProposal(
        model,
        output=output,
        poolsize=10,
        rescale_parameters=False,
    )
    fp.set_rescaling()

    n = 10
    x = model.new_point(n)

    x_prime, log_j = fp.rescale(x)

    x_recon, log_j_inv = fp.inverse_rescale(x_prime)

    assert len(log_j) == n
    assert (log_j == 0.0).all()
    np.testing.assert_array_equal(log_j, -log_j_inv)

    assert_structured_arrays_equal(x_prime, x)
    assert_structured_arrays_equal(x_recon, x_prime)
