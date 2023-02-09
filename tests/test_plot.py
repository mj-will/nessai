# -*- coding: utf-8 -*-
"""
Testing the plotting functions.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from nessai import plot
from nessai import config


@pytest.fixture()
def live_points(model):
    """Set of live points"""
    return model.new_point(N=100)


@pytest.fixture()
def live_points_1(model):
    """Second set of live points"""
    return model.new_point(N=200)


@pytest.fixture()
def nested_samples(live_points):
    return np.sort(live_points, order="logL")


@pytest.fixture(autouse=True)
def auto_close_figures():
    """Automatically close all figures after each test"""
    yield
    plt.close("all")


@pytest.mark.parametrize("line_styles", [True, False])
def test_nessai_style_enabled(line_styles):
    """Assert the style is applied with config.plotting.disable_style=False

    Tests with `line_styles` True and False
    """

    def func(a, b):
        return a + b

    with patch("nessai.plot.config.plotting.disable_style", False), patch(
        "seaborn.axes_style"
    ) as mock_style, patch("matplotlib.rc_context") as mock_rc:
        out = plot.nessai_style(line_styles=line_styles)(func)(1, 2)
    assert out == 3
    mock_style.assert_called_with("ticks")
    mock_rc.assert_called_once()
    d = mock_rc.call_args[0][0]["axes.prop_cycle"].by_key()
    if line_styles:
        d["linestyle"] == config.plotting.line_styles
        d["color"] == config.plotting.line_colours
    else:
        assert "linestyle" not in d


def test_nessai_style_disabled():
    """
    Assert the style isn't applied with config.plotting.disable_style=True
    """

    def func(a, b):
        return a + b

    with patch("nessai.plot.config.plotting.disable_style", True), patch(
        "seaborn.axes_style"
    ) as mock_style:
        out = plot.nessai_style(func)(1, 2)
    assert out == 3
    mock_style.assert_not_called()


@pytest.mark.parametrize("line_styles", [True, False])
@pytest.mark.integration_test
def test_nessai_style_integration(line_styles):
    """Assert the line colours and styles are applied correctly"""

    def func():
        return (
            plt.rcParams["axes.prop_cycle"].by_key().get("color"),
            plt.rcParams["axes.prop_cycle"].by_key().get("linestyle", None),
        )

    colours, line_styles = plot.nessai_style(line_styles=line_styles)(func)()
    assert colours == config.plotting.line_colours
    if line_styles:
        assert line_styles == config.plotting.line_styles
    else:
        assert line_styles is None

    # Assert rcParams are still set to the defaults
    defaults = mpl.rcParamsDefault
    # Set backend manually
    defaults["backend"] = plt.rcParams["backend"]
    assert plt.rcParams == mpl.rcParamsDefault


@pytest.mark.parametrize("bounds", [None, True])
def test_plot_live_points_bounds(live_points, bounds, model):
    """Test generating a plot for a set of live points."""
    if bounds:
        bounds = model.bounds
    fig = plot.plot_live_points(live_points, bounds=bounds)
    assert fig is not None
    plt.close()


@pytest.mark.parametrize("c", [None, "x"])
def test_plot_live_points_hue(live_points, c, model):
    """Test generating a plot for a set of live points with a hue."""
    fig = plot.plot_live_points(live_points, c=c)
    assert fig is not None
    plt.close()


def test_plot_live_points_constant_hue(live_points):
    """Test to make sure that constant hue is handled correctly"""
    live_points["logL"] = np.ones(live_points.size)
    fig = plot.plot_live_points(live_points, c="logL")
    assert fig is not None
    plt.close()


@pytest.mark.parametrize("save", [False, True])
def test_plot_live_points_save(live_points, save, model, tmpdir):
    """Test generating a plot for a set of live points and saving it."""
    if save:
        filename = os.path.join(tmpdir, "lp.png")
    else:
        filename = None
    plot.plot_live_points(live_points, filename=filename)
    plt.close()


def test_plot_live_points_1d():
    """Test generating the live points plot for one parameter"""
    live_points = np.random.randn(100).view([("x", "f8")])
    fig = plot.plot_live_points(live_points)
    assert fig is not None
    plt.close()


def test_plot_live_points_with_nans(live_points):
    """Assert a plot is made if one of parameters is all NaNs"""
    live_points["x"] = np.nan
    fig = plot.plot_live_points(live_points)
    assert fig is not None
    plt.close()


@pytest.mark.parametrize("parameters", [None, ["x", "y"]])
def test_plot_1d_comparison_unstructured(parameters):
    """Test plotting live points in arrays are not structured."""
    live_points = np.random.randn(10, 2)
    plot.plot_1d_comparison(
        live_points, convert_to_live_points=True, parameters=parameters
    )
    plt.close()


def test_plot_1d_comparison_unstructured_missing_flag():
    """Test plotting live points in arrays are not structured."""
    live_points = np.random.randn(10, 2)
    with pytest.raises(RuntimeError) as excinfo:
        plot.plot_1d_comparison(live_points, convert_to_live_points=False)

    assert "not structured array" in str(excinfo.value)


@pytest.mark.parametrize("parameters", [None, ["x", "y"]])
def test_plot_1d_comparison_parameters(parameters, live_points, live_points_1):
    """Test generating a 1d comparison plot"""
    plot.plot_1d_comparison(live_points, live_points_1, parameters=parameters)
    plt.close()


@pytest.mark.parametrize("save", [False, True])
def test_plot_1d_comparison_save(save, live_points, live_points_1, tmpdir):
    """Test generating a 1d comparison plot"""
    if save:
        filename = os.path.join(tmpdir, "comp.png")
    else:
        filename = None
    plot.plot_1d_comparison(live_points, live_points_1, filename=filename)
    plt.close()


@pytest.mark.parametrize("labels", [None, True])
def test_plot_1d_comparison_labels(labels, live_points, live_points_1, model):
    """Test generating a 1d comparison plot"""
    if labels:
        labels = model.names
    plot.plot_1d_comparison(live_points, live_points_1, labels=labels)
    plt.close()


def test_plot_1d_comparison_invalid_labels_list(live_points, live_points_1):
    """Assert an error is raised if the colours list is a different length."""
    with pytest.raises(ValueError) as excinfo:
        plot.plot_1d_comparison(
            live_points, live_points_1, labels=["r", "g", "b"]
        )
    assert "Length of labels list must match " in str(excinfo.value)


@pytest.mark.parametrize("bounds", [None, True])
def test_plot_1d_comparison_bounds(bounds, live_points, live_points_1, model):
    """Test generating a 1d comparison plot"""
    if bounds:
        bounds = model.bounds
    plot.plot_1d_comparison(live_points, live_points_1, bounds=bounds)
    plt.close()


def test_plot_1d_comparison_1d():
    """Test generating a 1d comparison plot with only one parameter"""
    l1 = np.random.randn(100).view([("x", "f8")])
    l2 = np.random.randn(100).view([("x", "f8")])
    plot.plot_1d_comparison(l1, l2)
    plt.close()


def test_plot_1d_comparison_nans(live_points):
    """Assert parameters containing only NaNs don't raise an error"""
    live_points["logL"] = np.nan
    live_points["logP"] = np.nan
    plot.plot_1d_comparison(live_points)
    plt.close()


def test_plot_1d_comparison_infinite_var(live_points, caplog):
    """Test generating a 1d comparirson with a variable that is not finite."""
    live_points["y"] = np.inf * np.ones(live_points.size)
    plot.plot_1d_comparison(live_points)
    plt.close()
    assert "No finite points for y, skipping." in str(caplog.text)


def test_plot_1d_comparison_infinite_var_comp(
    live_points, live_points_1, caplog
):
    """Test generating a 1d comparirson with a variable that is not finite but
    the second set of points contains finite values.
    """
    live_points["y"] = np.inf * np.ones(live_points.size)
    plot.plot_1d_comparison(live_points, live_points_1)
    plt.close()
    assert "No finite points for y, skipping." not in str(caplog.text)


@pytest.mark.parametrize("colours", [None, ["r", "g"]])
def test_plot_1d_comparison_colours(colours, live_points, live_points_1):
    """Test generating a 1d comparison plot with only one parameter"""
    plot.plot_1d_comparison(live_points, live_points_1, colours=colours)
    plt.close()


def test_plot_1d_comparison_more_colours(model):
    """Test generating a 1d comparirson when comparing more than 10 sets
    of live points.

    The default colour palettes all use 10 colours.
    """
    points = [model.new_point(N=10) for _ in range(12)]
    plot.plot_1d_comparison(*points)
    plt.close()


def test_plot_1d_comparison_invalid_colours_list(live_points, live_points_1):
    """Assert an error is raised if the colours list is a different length."""
    with pytest.raises(ValueError) as excinfo:
        plot.plot_1d_comparison(
            live_points, live_points_1, colours=["r", "g", "b"]
        )
    assert "Length of colours list must match " in str(excinfo.value)


@pytest.mark.parametrize("plot_breakdown", [False, True])
def test_plot_indices_breakdown(plot_breakdown):
    """Test plotting insertion indices with and without breakdown"""
    nlive = 100
    indices = np.random.randint(0, nlive, 1000)
    plot.plot_indices(indices, nlive=nlive, plot_breakdown=plot_breakdown)
    plt.close()


@pytest.mark.parametrize("save", [False, True])
def test_plot_indices_save(save, tmpdir):
    """Test plotting insertion indices with and without saving"""
    nlive = 100
    indices = np.random.randint(0, nlive, 1000)
    if save:
        filename = os.path.join(tmpdir, "indices.png")
    else:
        filename = None
    plot.plot_indices(indices, nlive=nlive, filename=filename)
    plt.close()


def test_plot_indices_no_indices():
    """Test to ensure plotting does not fail if indices are empty"""
    plot.plot_indices([], nlive=100)
    plt.close()


@pytest.mark.parametrize("sym", [False, True])
def test_plot_loss(sym):
    """Test function for plotting the loss"""
    epoch = 10
    if sym:
        history = dict(loss=-np.arange(epoch), val_loss=-np.arange(epoch))
    else:
        history = dict(loss=np.arange(epoch), val_loss=np.arange(epoch))
    plot.plot_loss(epoch, history)
    plt.close()


@pytest.mark.parametrize("save", [False, True])
def test_plot_loss_save(save, tmpdir):
    """Test function for plotting the loss"""
    epoch = 10
    history = dict(loss=np.arange(epoch), val_loss=np.arange(epoch))
    if save:
        filename = os.path.join(tmpdir, "loss.png")
    else:
        filename = None
    plot.plot_loss(epoch, history, filename=filename)
    plt.close()


@pytest.mark.parametrize("save", [False, True])
def test_trace_plot_save(nested_samples, save, tmpdir):
    """Test trace plot generation saving."""
    log_x = np.linspace(-10, 0, nested_samples.size)
    if save:
        filename = tmpdir + "trace.png"
    else:
        filename = None
    plot.plot_trace(log_x, nested_samples, filename=filename)
    plt.close()


def test_trace_plot_1d(nested_samples):
    """Test trace plot with only one parameter"""
    log_x = np.linspace(-10, 0, nested_samples.size)
    plot.plot_trace(log_x, nested_samples[["x"]])
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
    assert "structured array" in str(excinfo.value)


@pytest.mark.parametrize(
    "labels", [None, ["x", "y"] + config.livepoints.non_sampling_parameters]
)
def test_trace_plot_labels(nested_samples, labels):
    """Test trace plot generation with labels."""
    log_x = np.linspace(-10, 0, nested_samples.size)
    plot.plot_trace(log_x, nested_samples, labels=labels)
    plt.close()


def test_trace_plot_labels_error(nested_samples):
    """Test to ensure error is raised if labels are incompatible"""
    log_x = np.linspace(-10, 0, nested_samples.size)
    with pytest.raises(
        RuntimeError, match=r"List of labels is the wrong length \(3\)"
    ):
        plot.plot_trace(log_x, nested_samples, labels=["1", "2", "3"])


def test_trace_plot_parameters(nested_samples):
    """Assert the parameters arguments works"""
    log_x = np.linspace(-10, 0, nested_samples.size)
    plot.plot_trace(log_x, nested_samples, parameters=["x"])
    plt.close()


def test_trace_plot_kwargs(nested_samples):
    """Assert the kwargs are passed to the plotting function."""
    log_x = np.linspace(-10, 0, nested_samples.size)
    mock_axes = MagicMock()
    mock_axes.plot = MagicMock()
    with patch("matplotlib.pyplot.subplots", return_value=(None, mock_axes)):
        plot.plot_trace(log_x, nested_samples, marker="^", parameters=["x"])
    mock_axes.plot.call_args[0][1] == dict(marker="^")
    plt.close()


def test_histogram_plot():
    """Test the basic histogram plot"""
    x = np.random.randn(100)
    with patch("matplotlib.pyplot.hist") as mocked_hist:
        plot.plot_histogram(x, bins=10, density=True, label="test")
    mocked_hist.assert_called_once_with(
        x, bins=10, density=True, histtype="step"
    )


def test_histogram_plot_save(tmpdir):
    """Test to make sure the figure is saved."""
    x = np.random.randn(100)
    filename = os.path.join(tmpdir, "hist.png")
    fig = plot.plot_histogram(x, filename=filename)
    assert fig is None
    assert os.path.exists(filename)


def test_corner_plot(live_points):
    """Test the corner plot."""
    fig = plot.corner_plot(live_points)
    assert fig is not None


def test_corner_plot_check_inputs(live_points):
    """Assert corner.corner is called with the correct inputs"""
    with patch("corner.corner") as mock_corner:
        plot.corner_plot(live_points, test_kwarg="a")

    assert len(mock_corner.call_args[0]) == 1
    kwargs = mock_corner.call_args[1]
    assert "truths" in kwargs
    assert "labels" in kwargs
    assert "test_kwarg" in kwargs
    assert "color" in kwargs


@pytest.mark.parametrize(
    "labels",
    [["x", "y"], ["x", "y"] + config.livepoints.non_sampling_parameters],
)
def test_corner_plot_w_labels(live_points, labels):
    """Test the corner plot with labels"""
    plot.corner_plot(live_points, labels=labels)


@pytest.mark.parametrize(
    "truths",
    [[0, 0], [0, 0, None, None, None], {"x": 0, "y": 0}],
)
def test_corner_plot_w_truths(live_points, truths):
    """Test the corner plot with truths"""
    plot.corner_plot(live_points, truths=truths)


def test_corner_plot_w_exclude(live_points):
    """Test the parameter is excluded"""
    fig = plot.corner_plot(live_points, exclude=["y"])
    assert len(fig.axes) == 1


def test_corner_plot_w_include(live_points):
    """Test the parameter is included"""
    fig = plot.corner_plot(live_points, include=["x"])
    assert len(fig.axes) == 1


def test_corner_plot_w_include_and_labels(live_points):
    """Test the parameter is included and the labels do not raise an error"""
    fig = plot.corner_plot(live_points, include=["x"], labels=["x_0"])
    assert len(fig.axes) == 1


@pytest.mark.parametrize("truths", [[1], {"x": 1, "y": 1}])
def test_corner_plot_w_include_and_truths(live_points, truths):
    """Test the parameter is included and the truths do not raise an error"""
    fig = plot.corner_plot(live_points, include=["x"], truths=truths)
    assert len(fig.axes) == 1


def test_corner_plot_w_include_and_truths_error(live_points):
    """Assert an error is raised when the number truths do not match the
    number of parameters
    """
    with pytest.raises(ValueError, match=r"truths does not match .*"):
        plot.corner_plot(live_points, include=["x"], truths=[1, 1])


def test_corner_plot_all_nans(caplog, live_points):
    """Test how NaNs are handled.

    Should skip a parameter will al NaNs. In this case this should also include
    all of the non-sampling parameters since they are either NaNs or have zero
    dynamic range (it).
    """
    live_points["x"] = np.nan
    fig = plot.corner_plot(live_points)
    assert fig is not None
    assert (
        str(["x"] + config.livepoints.non_sampling_parameters) in caplog.text
    )


def test_corner_plot_save(tmpdir, live_points):
    """Assert the corner plot is saved."""
    filename = os.path.join(tmpdir, "corner.png")
    fig = plot.corner_plot(live_points, filename=filename)
    assert fig is None
    assert os.path.exists(filename)


def test_corner_plot_fields_exclude_error(live_points):
    """Assert an error is raised if include and exclude are both specified"""
    with pytest.raises(ValueError) as excinfo:
        plot.corner_plot(live_points, include=["x"], exclude=["logL"])
    assert "Cannot specify both `include` and `exclude`" in str(excinfo.value)
