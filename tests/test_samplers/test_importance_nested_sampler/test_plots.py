import matplotlib.pyplot as plt
from nessai.samplers.importancesampler import ImportanceNestedSampler as INS
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def auto_close_figures():
    """Automatically close all figures after each test"""
    yield
    plt.close("all")


def test_plot_state(ins, history, n_it):
    ins.iteration = n_it
    ins.history = history
    ins.history["checkpoint_iterations"] = [3, 4]
    ins.importance = dict(
        total=np.arange(-1, n_it),
        evidence=np.arange(-1, n_it),
        posterior=np.arange(-1, n_it),
    )
    ins.stopping_criterion = ["ratio", "ess"]
    ins.tolerance = [0.0, 1000]
    fig = INS.plot_state(ins)
    assert fig is not None


def test_plot_extra_state(ins, history, n_it):
    ins.iteration = n_it
    ins.checkpoint_iterations = [3, 4]
    ins.history = history
    fig = INS.plot_extra_state(ins)
    assert fig is not None


@pytest.mark.parametrize("enable_colours", [False, True])
def test_plot_trace(ins, samples, enable_colours):
    ins.samples_unit = samples
    fig = INS.plot_trace(ins, enable_colours=enable_colours)
    assert fig is not None


def test_plot_likelihood_levels(ins, samples):
    ins.samples_unit = samples
    fig = INS.plot_likelihood_levels(ins)
    assert fig is not None


def test_plot_level_cdf(ins, samples):
    cdf = np.cumsum(samples["logW"])
    fig = INS.plot_level_cdf(
        ins,
        samples["logL"],
        cdf,
        q=0.5,
        threshold=samples[len(samples) // 2]["logL"],
    )
    assert fig is not None
