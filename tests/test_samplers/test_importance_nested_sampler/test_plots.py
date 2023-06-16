import matplotlib.pyplot as plt
from nessai.samplers.importancesampler import ImportanceNestedSampler as INS
import pytest


@pytest.fixture(autouse=True)
def auto_close_figures():
    """Automatically close all figures after each test"""
    yield
    plt.close("all")


@pytest.mark.parametrize("enable_colours", [False, True])
def test_plot_trace(ins, samples, enable_colours):
    ins.samples = samples
    fig = INS.plot_trace(ins, enable_colours=enable_colours)
    assert fig is not None


def test_plot_likelihood_levels(ins, samples):
    ins.samples = samples
    fig = INS.plot_likelihood_levels(ins)
    assert fig is not None
