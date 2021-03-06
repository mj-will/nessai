# -*- coding: utf-8 -*-
"""
Testing the plotting functions.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pytest
from unittest.mock import patch

from nessai import plot
from nessai.livepoint import numpy_array_to_live_points


@pytest.fixture()
def live_points(model):
    """Set of live points"""
    return model.new_point(N=100)


@pytest.fixture()
def nested_samples(live_points):
    return np.sort(live_points, order='logL')


@pytest.mark.parametrize('bounds', [None, True])
def test_plot_live_points_bounds(live_points, bounds, model):
    """Test generating a plot for a set of live points."""
    if bounds:
        bounds = model.bounds
    plot.plot_live_points(live_points, bounds=bounds)


@pytest.mark.parametrize('c', [None, 'x'])
def test_plot_live_points_hue(live_points, c, model):
    """Test generating a plot for a set of live points with a hue."""
    plot.plot_live_points(live_points, c=c)
    plt.close()


@pytest.mark.parametrize('save', [False, True])
def test_plot_live_points_save(live_points, save, model, tmpdir):
    """Test generating a plot for a set of live points and saving it."""
    if save:
        filename = tmpdir + 'lp.png'
    else:
        filename = None
    plot.plot_live_points(live_points, filename=filename)
    plt.close()


def test_plot_live_points_1d():
    """Test generating the live points plot for one parameter"""
    live_points = np.random.randn(100).view([('x', 'f8')])
    plot.plot_live_points(live_points)
    plt.close()


def test_plot_live_points_error(live_points):
    """
    Test to make sure that an error is not raised if plotting fails
    because of issues with seaborn and gwpy.
    """
    with patch('seaborn.scatterplot', side_effect=TypeError('Mock error')):
        fig = plot.plot_live_points(live_points)
    assert fig is None


@pytest.mark.parametrize('parameters', [None, ['x', 'y']])
def test_plot_1d_comparison_unstructured(parameters):
    """Test plotting live points in arrays are not structured."""
    live_points = np.random.randn(100, 2)
    plot.plot_1d_comparison(live_points, convert_to_live_points=True,
                            parameters=parameters)
    plt.close()


def test_plot_1d_comparison_unstructured_missing_flag():
    """Test plotting live points in arrays are not structured."""
    live_points = np.random.randn(100, 2)
    with pytest.raises(RuntimeError) as excinfo:
        plot.plot_1d_comparison(live_points, convert_to_live_points=False)

    assert 'not structured array' in str(excinfo.value)


@pytest.mark.parametrize('parameters', [None, ['x', 'y']])
def test_plot_1d_comparison_parameters(parameters, model):
    """Test generating a 1d comparison plot"""
    l1 = model.new_point(N=100)
    l2 = model.new_point(N=100)
    plot.plot_1d_comparison(l1, l2, parameters=parameters)
    plt.close()


@pytest.mark.parametrize('save', [False, True])
def test_plot_1d_comparison_save(save, model, tmpdir):
    """Test generating a 1d comparison plot"""
    l1 = model.new_point(N=100)
    l2 = model.new_point(N=100)
    if save:
        filename = tmpdir + 'comp.png'
    else:
        filename = None
    plot.plot_1d_comparison(l1, l2, filename=filename)
    plt.close()


@pytest.mark.parametrize('labels', [None, True])
def test_plot_1d_comparison_labels(labels, model):
    """Test generating a 1d comparison plot"""
    l1 = model.new_point(N=100)
    l2 = model.new_point(N=100)
    if labels:
        labels = model.names
    plot.plot_1d_comparison(l1, l2, labels=labels)
    plt.close()


@pytest.mark.parametrize('bounds', [None, True])
def test_plot_1d_comparison_bounds(bounds, model):
    """Test generating a 1d comparison plot"""
    l1 = model.new_point(N=100)
    l2 = model.new_point(N=100)
    if bounds:
        bounds = model.bounds
    plot.plot_1d_comparison(l1, l2, bounds=bounds)
    plt.close()


def test_plot_1d_comparison_1d():
    """Test generating a 1d comparison plot with only one parameter"""
    l1 = np.random.randn(100).view([('x', 'f8')])
    l2 = np.random.randn(100).view([('x', 'f8')])
    plot.plot_1d_comparison(l1, l2)
    plt.close()


@pytest.mark.parametrize('plot_breakdown', [False, True])
def test_plot_indices_breakdown(plot_breakdown):
    """Test plotting insertion indices with and without breakdown"""
    nlive = 100
    indices = np.random.randint(0, nlive, 1000)
    plot.plot_indices(indices, nlive=nlive, plot_breakdown=plot_breakdown)
    plt.close()


@pytest.mark.parametrize('save', [False, True])
def test_plot_indices_save(save, tmpdir):
    """Test plotting insertion indices with and without saving"""
    nlive = 100
    indices = np.random.randint(0, nlive, 1000)
    if save:
        filename = tmpdir + 'indices.png'
    else:
        filename = None
    plot.plot_indices(indices, nlive=nlive, filename=filename)
    plt.close()


def test_plot_indices_no_indices():
    """Test to ensure plotting does not fail if indices are empty"""
    plot.plot_indices([], nlive=100)
    plt.close()


@pytest.mark.parametrize('sym', [False, True])
def test_plot_loss(sym):
    """Test function for plotting the loss"""
    epoch = 10
    if sym:
        history = dict(loss=-np.arange(epoch),
                       val_loss=-np.arange(epoch))
    else:
        history = dict(loss=np.arange(epoch), val_loss=np.arange(epoch))
    plot.plot_loss(epoch, history)
    plt.close()


@pytest.mark.parametrize('save', [False, True])
def test_plot_loss_save(save, tmpdir):
    """Test function for plotting the loss"""
    epoch = 10
    history = dict(loss=np.arange(epoch), val_loss=np.arange(epoch))
    if save:
        filename = tmpdir + 'loss.png'
    else:
        filename = None
    plot.plot_loss(epoch, history, filename=filename)
    plt.close()


@pytest.mark.parametrize('save', [False, True])
def test_trace_plot_save(nested_samples, save, tmpdir):
    """Test trace plot generation saving."""
    log_x = np.linspace(-10, 0, nested_samples.size)
    if save:
        filename = tmpdir + 'trace.png'
    else:
        filename = None
    plot.plot_trace(log_x, nested_samples, filename=filename)
    plt.close()


def test_trace_plot_1d(nested_samples):
    """Test trace plot with only one parameter"""
    log_x = np.linspace(-10, 0, nested_samples.size)
    nested_samples = numpy_array_to_live_points(
        np.random.randn(log_x.size, 1), ['x'])
    plot.plot_trace(log_x, nested_samples)
    plt.close()


def test_trace_plot_unstructured():
    """
    Test to check that trace_plot raises an error when the nested samples
    are not a structured array.
    """
    log_x = np.linspace(-10, 0, 100)
    nested_samples = np.random.randn(log_x.size, 2)
    with pytest.raises(TypeError) as excinfo:
        plot.plot_trace(log_x, nested_samples)
    plt.close()
    assert 'structured array' in str(excinfo.value)


@pytest.mark.parametrize('labels', [None, ['x', 'y']])
def test_trace_plot_labels(nested_samples, labels, tmpdir):
    """Test trace plot generation with labels."""
    log_x = np.linspace(-10, 0, nested_samples.size)
    plot.plot_trace(log_x, nested_samples, labels=labels)
    plt.close()


def test_trace_plot_labels_error(nested_samples):
    """Test to ensure error is raised if labels are incompatible"""
    log_x = np.linspace(-10, 0, nested_samples.size)
    with pytest.raises(RuntimeError) as excinfo:
        plot.plot_trace(log_x, nested_samples, labels=['1', '2', '3'])

    assert 'Missing labels' in str(excinfo.value)


def test_histogram_plot():
    """Test the basic histogram plot"""
    x = np.random.randn(100)
    with patch('matplotlib.pyplot.hist') as mocked_hist:
        plot.plot_histogram(x, bins=10, density=True, label='test')
    mocked_hist.assert_called_once_with(
        x, bins=10, density=True, histtype='step')


def test_histogram_plot_save(tmpdir):
    """Test to make sure the figure is saved."""
    x = np.random.randn(100)
    fig = plot.plot_histogram(x, filename=tmpdir + 'hist.png')
    assert fig is None
    assert os.path.exists(tmpdir + 'hist.png')
