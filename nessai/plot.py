# -*- coding: utf-8 -*-
"""
Plotting utilities.
"""
import functools
import logging

from cycler import cycler
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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
            with sns.axes_style(config.plotting.sns_style), mpl.rc_context(
                {**_rcparams, **d}
            ):
                return func(*args, **kwargs)

        return wrapper

    if callable(line_styles):
        return decorator(line_styles)
    else:
        return decorator


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

    fig, axs = plt.subplots(
        len(parameters), 1, sharey=False, figsize=(3, 3 * len(parameters))
    )

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
        handles, labels = plt.gca().get_legend_handles_labels()
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

    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


@nessai_style(line_styles=False)
def plot_indices(indices, nlive=None, filename=None, plot_breakdown=True):
    """
    Histogram indices for index insertion tests, also includes the CDF.

    Parameters
    ----------
    indices : array_like
        List of insertion indices to plot
    nlive : int
        Number of live points used in the nested sampling run
    filename : str
        Filename used to save the figure.
    plot_breakdown : bool, optional
       If true, then the CDF for every nlive points is also plotted as grey
       lines.
    """
    indices = np.asarray(indices)
    if not indices.size or not nlive:
        logger.warning("Not producing indices plot.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    nbins = min(len(np.histogram_bin_edges(indices, "auto")) - 1, 1000)
    if plot_breakdown:
        for i in range(len(indices) // nlive):
            ax[1].hist(
                indices[i * nlive : (i + 1) * nlive],
                bins=nlive,
                histtype="step",
                density=True,
                alpha=0.1,
                color="black",
                lw=0.5,
                cumulative=True,
                range=(0, nlive - 1),
            )

    ax[0].hist(
        indices,
        density=True,
        color="tab:blue",
        linewidth=1.25,
        histtype="step",
        bins=nbins,
        label="produced",
        range=(0, nlive - 1),
    )
    ax[1].hist(
        indices,
        density=True,
        color="tab:blue",
        linewidth=1.25,
        histtype="step",
        bins=nlive,
        label="produced",
        cumulative=True,
        range=(0, nlive - 1),
    )

    if nlive is not None:
        ax[0].axhline(
            1 / nlive,
            color="black",
            linewidth=1.25,
            linestyle="-",
            label="pmf",
            alpha=0.5,
        )
        ax[0].axhline(
            (1 + (nbins / len(indices)) ** 0.5) / nlive,
            color="black",
            linewidth=1.25,
            linestyle=":",
            alpha=0.5,
            label="1-sigma",
        )
        ax[0].axhline(
            (1 - (nbins / len(indices)) ** 0.5) / nlive,
            color="black",
            linewidth=1.25,
            linestyle=":",
            alpha=0.5,
        )
        ax[1].plot(
            [0, nlive],
            [0, 1],
            color="black",
            linewidth=1.25,
            linestyle=":",
            label="cmf",
        )

    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower right")
    ax[0].set_xlim([0, nlive - 1])
    ax[1].set_xlim([0, nlive - 1])
    ax[0].set_xlabel("Insertion indices")
    ax[1].set_xlabel("Insertion indices")

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
    log_x,
    nested_samples,
    parameters=None,
    labels=None,
    filename=None,
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

    fig, axes = plt.subplots(
        len(labels), 1, figsize=(5, 3 * len(labels)), sharex=True
    )
    if len(labels) > 1:
        axes = axes.ravel()
    else:
        axes = [axes]

    for i, name in enumerate(parameters):
        axes[i].plot(log_x, nested_samples[name], **default_kwargs)
        axes[i].set_ylabel(labels[i])

    axes[-1].set_xlabel("log X")
    axes[-1].invert_xaxis()

    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
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
        quantiles=[0.16, 0.84],
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

    if truths:
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
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig
