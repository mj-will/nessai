# -*- coding: utf-8 -*-
"""
Plotting utilities.
"""

import functools
import logging
from typing import List, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib import pyplot as plt

from . import config
from .livepoint import live_points_to_array
from .utils import auto_bins

logger = logging.getLogger(__name__)

_rcparams = sns.plotting_context("notebook")
_rcparams.update(
    {
        "legend.frameon": False,
    }
)


def nessai_style(line_styles=True):
    """Decorator for plotting function that sets the style.

    Functions as both standard decorator :code:`@nessai_style` or as a callable
    decorator :code:`@nessai_style()`.

    Style can be disabled by setting :py:data:`nessai.config.DISABLE_STYLE` to
    :code:`True`.

    Parameters
    ----------
    line_styles : boolean
        Use custom line styles.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if config.plotting.disable_style:
                return func(*args, **kwargs)
            c = cycler(color=config.plotting.line_colours)
            if line_styles:
                c += cycler(linestyle=config.plotting.line_styles)
            d = {
                "axes.prop_cycle": c,
            }
            with (
                sns.axes_style(config.plotting.sns_style),
                mpl.rc_context({**_rcparams, **d}),
            ):
                return func(*args, **kwargs)

        return wrapper

    if callable(line_styles):
        return decorator(line_styles)
    else:
        return decorator


def sanitise_array(
    a: np.ndarray,
    /,
    a_min: Optional[float] = None,
    a_max: Optional[float] = None,
):
    """Sanitise an array for plotting.

    If :code:`x_min` is not specified, it is set to the value in
    :code:`nessai.config.plotting.clip_min`.

    Parameters
    ----------
    x : array_like
        Array to sanitise.
    x_min : float, optional
        Minimum value to clip the data to.
    xmax : float, optional
        Maximum value to clip the data to.

    Returns
    -------
    np.ndarray
        Sanitised array.
    """
    if a_min is None:
        a_min = config.plotting.clip_min
    return np.clip(a, a_min, a_max)


@nessai_style()
def plot_live_points(
    live_points, filename=None, bounds=None, c=None, **kwargs
):
    """
    Plot a set of live points in a corner-like plot.

    Will drop columns where all elements are NaNs

    Parameters
    ----------
    live_points : ndarray
        Structured array of live points to plot.
    filename : str
        Filename for resulting figure
    bounds : dict:
        Dictionary of lower and upper bounds to plot
    c : str, optional
        Name of field in the structured array to use as the hue when plotting
        the samples. If not specified, no hue is used.
    kwargs :
        Keyword arguments used to update the pairplot kwargs. Diagonal and off-
        diagonal plots can be configured with ``diag_kws`` and ``plot_kws``.
    """
    pairplot_kwargs = dict(
        corner=True,
        kind="scatter",
        diag_kws=dict(
            histtype="step",
            bins="auto",
            lw=1.5,
            density=True,
            color=config.plotting.base_colour,
        ),
        plot_kws=dict(
            s=1.0,
            edgecolor=None,
            palette="viridis",
            color=config.plotting.base_colour,
        ),
    )
    pairplot_kwargs.update(kwargs)

    df = pd.DataFrame(live_points)
    df = df.dropna(axis="columns", how="all")
    df = df[np.isfinite(df).all(1)]

    if c is not None:
        hue = df[c]
        if np.all(hue == hue[0]):
            logger.warning(
                f"Selected hue variable: {c} is constant! Disabling."
            )
            hue = None
    else:
        hue = None

    if hue is None:
        pairplot_kwargs["plot_kws"].pop("palette")

    fig = sns.PairGrid(df, corner=True, diag_sharey=False)
    fig.map_diag(plt.hist, **pairplot_kwargs["diag_kws"])
    fig.map_offdiag(sns.scatterplot, hue=hue, **pairplot_kwargs["plot_kws"])

    if bounds is not None:
        for i, v in enumerate(bounds.values()):
            fig.axes[i, i].axvline(
                v[0],
                ls=":",
                alpha=0.5,
                color=config.plotting.highlight_colour,
            )
            fig.axes[i, i].axvline(
                v[1],
                ls=":",
                alpha=0.5,
                color=config.plotting.highlight_colour,
            )

    if filename is not None:
        fig.savefig(filename)
        plt.close()
    else:
        return fig


@nessai_style()
def plot_1d_comparison(
    *live_points,
    parameters=None,
    labels=None,
    colours=None,
    bounds=None,
    hist_kwargs={},
    filename=None,
    convert_to_live_points=False,
):
    """
    Plot 1d histograms comparing different sets of live points

    Parameters
    ----------
    *live_points : iterable of ndarrays
        Variable length argument of live points in structured arrays with
        fields. Also see ``parameters`` argument.
    parameters : array_like, optional
        Array of parameters (field names) to plot. Default None implies all
        fields are plotted.
    labels : list, optional
        List of labels for each structured array being plotted (default None).
        If None each set of live points is labelled numerically
    colours : list, optional
        List of colours to use for each set of live points.
    bounds : dict, optional
        Dictionary of upper and lowers bounds to plot. Each key must
        match a field and each value must be an interable of length 2 in order
        lower then upper bound. If None (default), no bounds plotted.
    hist_kwargs : dict, optional
        Dictionary of keyword arguments passed to matplotlib.pyplot.hist.
    filename : str, optional
        Name of file for saving figure. (Default None implies figure is not
        saved).
    convert_to_live_points : bool, optional
        Set to true if inputs are not structured arrays of live points
    """
    if convert_to_live_points:
        live_points = list(live_points)
        if parameters is None:
            parameters = [i for i in range(live_points[0].shape[-1])]
        for i in range(len(live_points)):
            live_points[i] = {
                k: v for k, v in zip(parameters, live_points[i].T)
            }

    elif any(lp.dtype.names is None for lp in live_points):
        raise RuntimeError(
            "Live points are not structured arrays"
            "Set `convert_to_live_points=True`."
        )
    elif parameters is None:
        parameters = live_points[0].dtype.names

    if labels is None:
        labels = [str(i) for i in range(len(live_points))]
    elif not len(labels) == len(live_points):
        raise ValueError(
            "Length of labels list must match number of arrays being plotted."
        )

    if colours is None:
        colours = sns.color_palette()
        colours = int(np.ceil(len(live_points) / len(colours))) * colours
    elif not len(colours) == len(live_points):
        raise ValueError(
            "Length of colours list must match number of arrays being plotted."
        )

    figsize = (3, min(config.plotting.max_figsize, 3 * len(parameters)))
    fig, axs = plt.subplots(len(parameters), 1, sharey=False, figsize=figsize)

    if len(parameters) > 1:
        axs = axs.ravel()
    else:
        axs = [axs]

    for i, f in enumerate(parameters):
        finite_points = []
        include = []
        for j, lp in enumerate(live_points):
            if np.isnan(lp[f]).all():
                continue
            idx = np.isfinite(lp[f])
            if idx.any():
                finite_points.append(lp[f][idx])
                include.append(j)
        if not include:
            logger.warning(f"No finite points for {f}, skipping.")
            continue

        xmin = np.min([p.min() for p in finite_points])
        xmax = np.max([p.max() for p in finite_points])

        for j, p in enumerate(finite_points):
            orig_idx = include[j]
            axs[i].hist(
                p,
                bins=auto_bins(p),
                histtype="step",
                range=(xmin, xmax),
                density=True,
                label=labels[orig_idx],
                color=colours[orig_idx],
                **hist_kwargs,
            )
        axs[i].set_xlabel(f)
        if bounds is not None and f in bounds:
            axs[i].axvline(bounds[f][0], ls=":", alpha=0.5, color="k")
            axs[i].axvline(bounds[f][1], ls=":", alpha=0.5, color="k")

    if len(labels) > 1:
        # Get the handles
        # Some axes may have less due to e.g. all NaN values, so loop over them
        # all until a valid set of handles is found.
        for ax in axs:
            handles, _ = ax.get_legend_handles_labels()
            if len(handles) == len(labels):
                break
        if len(handles) == len(labels):
            legend_labels = dict(zip(labels, handles))
            fig.legend(
                legend_labels.values(),
                legend_labels.keys(),
                frameon=False,
                ncol=len(labels),
                loc="upper center",
                bbox_to_anchor=(0, 0.1 / len(parameters), 1, 1),
                bbox_transform=plt.gcf().transFigure,
            )
        else:
            logger.warning("Could not plot legend")

    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


@nessai_style(line_styles=False)
def plot_indices(
    indices,
    nlive=None,
    filename=None,
    ks_test_mode="D+",
    confidence_intervals=(0.68, 0.95, 0.997),
    plot_breakdown=True,
    n_breakdown=8,
    cmap="viridis",
):
    """
    Histogram indices for index insertion tests, also includes the cmf.

    Parameters
    ----------
    indices : array_like
        List of insertion indices to plot
    nlive : int
        Number of live points used in the nested sampling run. If nlive is not
        specified, it is set to the maximum value in indices.
    filename : str
        Filename used to save the figure.
    plot_breakdown : bool, optional
       If true, then the CDF for every nlive points is also plotted as grey
       lines.
    ks_test_mode : Literal["D+", "D-"]
        Mode for computing the KS test. See
        :py:func:`nessai.utils.indices.compute_indices_ks_test` for details.
    confidence_intervals : tuple
        Confidence intervals to plot as shaded regions on the cmf plot.
    plot_breakdown : bool
        If true, plots the cmf for batches of samples over the course of the
        run. The number of batches in controlled by :code:`n_breakdown`.
    n_breakdown : int
        The number of batches to plot. Also see :code:`plot_breakdown`.
    cmap : str
        Colourmap to use when :code:`plot_breakdown=True`.
    """
    from scipy import stats

    from .utils.indices import compute_indices_ks_test

    indices = np.asarray(indices)
    if not indices.size:
        logger.warning("Not producing indices plot.")
        return

    if nlive is None:
        logger.warning(
            "Estimating nlive from insertion indices. "
            "The reported p-value may be incorrect."
        )
        nlive = np.max(indices) + 1

    _, p_value = compute_indices_ks_test(indices, nlive, mode=ks_test_mode)

    # First bin should have non-zero probability since this is a p.m.f
    x = np.arange(1.0, nlive + 1, 1)
    analytic_cmf = x / x[-1]
    counts = np.bincount(indices, minlength=nlive)
    estimated_cmf = np.cumsum(counts) / len(indices)

    if plot_breakdown:
        n_cols = 3
        figsize = (15, 5)
    else:
        n_cols = 2
        figsize = (10, 5)

    fig, ax = plt.subplots(1, ncols=n_cols, figsize=figsize)
    nbins = min(len(np.histogram_bin_edges(indices, "auto")) - 1, 1000)

    # Plot the analytic p.m.f first
    ax[0].axhline(
        1 / nlive,
        color="black",
        linestyle="-",
        label="pmf",
        alpha=0.5,
    )
    # 1-sigma regions
    ax[0].axhline(
        (1 + (nbins / len(indices)) ** 0.5) / nlive,
        color="black",
        linestyle=":",
        alpha=0.5,
        label="1-sigma",
    )
    ax[0].axhline(
        (1 - (nbins / len(indices)) ** 0.5) / nlive,
        color="black",
        linestyle=":",
        alpha=0.5,
    )

    ax[0].hist(
        indices,
        density=True,
        color="C0",
        histtype="step",
        bins=nbins,
        label="Estimated",
        range=(0, nlive - 1),
    )

    # Subtract 1 since we count indices from 0
    ax[1].plot(
        x - 1,
        analytic_cmf - estimated_cmf,
        c="C0",
        label="Analytic cmf - Estimated cmf",
    )
    n_indices = len(indices)
    for ci in confidence_intervals:
        bound = (1 - ci) / 2
        bound_values = (
            stats.binom.ppf(1 - bound, n_indices, analytic_cmf) / n_indices
        )
        lower = bound_values - analytic_cmf
        upper = analytic_cmf - bound_values

        ax[1].fill_between(x - 1, lower, upper, color="grey", alpha=0.2)

    ax[0].legend(loc="lower right")
    ax[0].set_xlim([0, nlive - 1])
    ax[0].set_xlabel("Insertion index")

    ax[1].legend(loc="lower right")
    ax[1].set_xlim([0, nlive - 1])
    ax[1].set_xlabel("Insertion index")

    if plot_breakdown:
        lw = 0.5 * plt.rcParams["lines.linewidth"]
        batches = np.array_split(indices, n_breakdown)
        colours = sns.color_palette(n_colors=n_breakdown, palette=cmap)
        for batch, colour in zip(batches, colours):
            counts = np.bincount(batch, minlength=nlive)
            batch_estimated_cmf = np.cumsum(counts) / len(batch)
            ax[2].plot(
                x - 1,
                analytic_cmf - batch_estimated_cmf,
                lw=lw,
                c=colour,
            )
        for ci in confidence_intervals:
            bound = (1 - ci) / 2
            bound_values = stats.binom.ppf(
                1 - bound, len(batch), analytic_cmf
            ) / len(batch)
            lower = bound_values - analytic_cmf
            upper = analytic_cmf - bound_values
            ax[2].fill_between(x - 1, lower, upper, color="grey", alpha=0.2)
        ax[2].set_xlim([0, nlive - 1])
        ax[2].set_xlabel("Insertion index")

    fig.suptitle(f"p-value={p_value:.4f} (nlive={nlive})")

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


@nessai_style()
def plot_loss(epoch, history, filename=None):
    """
    Plot the loss function per epoch.

    Parameters
    ----------
    epoch : int
        Final training epoch
    history : dict
        Dictionary with keys ``'loss'`` and ``'val_loss'``
    filename : str, optional
        Path for saving the figure. If not specified figure is returned
        instead.
    """
    fig, ax = plt.subplots()
    epochs = np.arange(1, epoch + 1, 1)
    plt.plot(epochs, history["loss"], label="Loss")
    plt.plot(epochs, history["val_loss"], label="Val. loss")
    plt.xlabel("Epochs")
    plt.ylabel("Negative log-likelihood")
    plt.legend()
    plt.tight_layout()
    if any(h < 0 for h in history["loss"]):
        plt.yscale("symlog")
    else:
        plt.yscale("log")

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


@nessai_style()
def plot_trace(
    log_x: np.ndarray,
    nested_samples: np.ndarray,
    parameters: Optional[List[str]] = None,
    live_points: Optional[np.ndarray] = None,
    log_x_live_points: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    filename: Optional[str] = None,
    **kwargs,
):
    """Produce trace plot for the nested samples.

    By default this includes all parameters in the samples, not just those
    included in the model being sampled.

    Parameters
    ----------
    log_x : array_like
        Array of log prior volumes
    nested_samples : ndarray
        Array of nested samples to plot
    parameters : list, optional
        List of parameters to include the trace plot. If not specified, all of
        the parameters in the nested samples are included.
    live_points : ndarray
        Optional array of live points to include in the plot. See also
        :code:`log_x_live_points`.
    log_x_live_points : ndarray
        Optional array of log-prior volumes for the live points. Required if
        :code:`live_points` is specified.
    labels : list, optional
        List of labels to use instead of the names of parameters
    filename : str, optional
        Filename for saving the plot, if none plot is not saved and figure
        is returned instead.
    kwargs :
        Keyword arguments passed to :code:`matplotlib.pyplot.plot`.
    """
    default_kwargs = dict(
        marker=",",
        linestyle="",
    )

    nested_samples = np.asarray(nested_samples)
    if not nested_samples.dtype.names:
        raise TypeError("Nested samples must be a structured array")

    if parameters is None:
        parameters = nested_samples.dtype.names
    if labels is None:
        labels = parameters

    if not len(labels) == len(parameters):
        raise RuntimeError(
            f"List of labels is the wrong length ({len(labels)}) for the "
            f"parameters: {parameters}."
        )
    if kwargs:
        default_kwargs.update(kwargs)

    figsize = (5, min(config.plotting.max_figsize, 3 * len(labels)))
    fig, axes = plt.subplots(len(labels), 1, figsize=figsize, sharex=True)
    if len(labels) > 1:
        axes = axes.ravel()
    else:
        axes = [axes]

    for i, name in enumerate(parameters):
        axes[i].plot(log_x, nested_samples[name], **default_kwargs)
        if live_points is not None:
            axes[i].plot(
                log_x_live_points, live_points[name], **default_kwargs
            )
        axes[i].set_ylabel(labels[i])

    axes[-1].set_xlabel("log X")
    axes[-1].invert_xaxis()

    if filename is not None:
        try:
            fig.savefig(filename, bbox_inches="tight")
        except ValueError as e:
            logger.warning(f"Could not save trace plot. Error: {e}")
        plt.close(fig)
    else:
        return fig


@nessai_style()
def plot_histogram(samples, label=None, filename=None, **kwargs):
    """Plot a histogram of samples.

    Parameters
    ----------
    samples : array_like
        Samples to plot.
    label : str, optional
        Label to the x axis.
    filename : str, optional
        Filename for saving the plot. If not specified, figure is returned.
    kwargs :
        Keyword arguments passed to `matplotlib.pyplot.hist`.
    """
    default_kwargs = dict(histtype="step", bins=auto_bins(samples))
    default_kwargs.update(kwargs)
    fig = plt.figure()
    plt.hist(samples, **default_kwargs)
    if label is not None:
        plt.xlabel(label)
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


@nessai_style()
def corner_plot(
    array,
    include=None,
    exclude=None,
    labels=None,
    truths=None,
    filename=None,
    **kwargs,
):
    """Produce a corner plot for a structured array.

    Removes any fields with no dynamic range.

    Parameters
    ----------
    array : numpy.ndarray
        Structured array
    include : Optional[list]
        List of parameters to plot.
    exclude : Optional[list]
        List of parameters to exclude.
    labels : Optional[Iterable]
        Labels for each parameter that is to be plotted.
    truths : Optional[Union[Iterable, Dict]]
        Truth values for each parameters, parameters can be skipped by setting
        the value to None.
    filename : Optional[str]
        Filename for saving the plot. If not specified, figure is returned.
    kwargs : Dict[Any]
        Dictionary of keyword arguments passed to :code:`corner.corner`.
    """
    import corner

    default_kwargs = dict(
        bins=32,
        smooth=0.9,
        color=config.plotting.base_colour,
        truth_color=config.plotting.highlight_colour,
        quantiles=[0.16, 0.5, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
        plot_density=True,
        plot_datapoints=True,
        fill_contours=True,
        show_titles=True,
        hist_kwargs=dict(density=True),
    )
    if kwargs:
        default_kwargs.update(kwargs)

    if include and exclude:
        raise ValueError("Cannot specify both `include` and `exclude`")

    if exclude:
        include = [n for n in array.dtype.names if n not in exclude]
    if include:
        array = array[include]
    if labels is None:
        labels = np.asarray(array.dtype.names)
    else:
        labels = np.asarray(labels)

    unstruct_array = live_points_to_array(array)

    has_range = np.array(
        [
            (~np.isnan(v).all()) and (~(np.nanmin(v) == np.nanmax(v)))
            for v in unstruct_array.T
        ],
        dtype=bool,
    )
    if not all(has_range):
        logger.warning(
            "Some parameters have no dynamic range. Removing: "
            f"{[n for n, b in zip(array.dtype.names, has_range) if not b]}"
        )
    unstruct_array = unstruct_array[..., has_range]

    if len(labels) != unstruct_array.shape[-1]:
        labels = labels[has_range]

    if truths is not None:
        if isinstance(truths, dict):
            if include:
                truths = np.array([truths[n] for n in include])
            else:
                truths = np.fromiter(truths.values(), float)
        else:
            truths = np.asarray(truths)
        if len(truths) != unstruct_array.shape[-1]:
            if not all(has_range):
                truths = truths[has_range]
            else:
                raise ValueError(
                    "Length of truths does not match number of "
                    "parameters being plotted"
                )

    fig = corner.corner(
        unstruct_array, truths=truths, labels=labels, **default_kwargs
    )

    if filename is not None:
        try:
            fig.savefig(filename, bbox_inches="tight")
        except ValueError as e:
            logger.warning(f"Could not save corner plot. Error: {e}")
        plt.close(fig)
    else:
        return fig
