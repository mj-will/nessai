# -*- coding: utf-8 -*-
"""
Plotting utilities.
"""
import logging
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from .utils import auto_bins

sns.set()
sns.set_style('ticks')

logger = logging.getLogger(__name__)

pairplot_kwargs = dict(corner=True, kind='scatter',
                       diag_kws=dict(histtype='step', bins='auto', lw=1.5,
                                     density=True, color='teal'),
                       plot_kws=dict(s=1.0, edgecolor=None, palette='viridis',
                                     color='teal'))


def plot_live_points(live_points, filename=None, bounds=None, c=None,
                     **kwargs):
    """
    Plot a set of live points in a corner-like plot.

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
    pairplot_kwargs.update(kwargs)

    df = pd.DataFrame(live_points)
    df = df[np.isfinite(df).all(1)]

    if c is not None:
        hue = df[c]
        if np.all(hue == hue[0]):
            logger.warning(
                f'Selected hue variable: {c} is constant! Disabling.')
            hue = None
    else:
        hue = None

    logger.debug(f'Producing corner-like plot for {len(df)} points.')

    try:
        fig = sns.PairGrid(df, corner=True, diag_sharey=False)
        fig.map_diag(plt.hist, **pairplot_kwargs['diag_kws'])
        fig.map_offdiag(sns.scatterplot, hue=hue,
                        **pairplot_kwargs['plot_kws'])
    except TypeError as e:
        plt.close()
        logger.warning('Could not produce a plot of the live points. '
                       'Check the version of Seaborn, Matplotlib and GWpy. '
                       f'The error was: \n {e}')
        return None

    if bounds is not None:
        for i, v in enumerate(bounds.values()):
            fig.axes[i, i].axvline(v[0], ls=':', alpha=0.5, color='k')
            fig.axes[i, i].axvline(v[1], ls=':', alpha=0.5, color='k')

    if filename is not None:
        fig.savefig(filename)
        plt.close()
    else:
        return fig


def plot_1d_comparison(*live_points, parameters=None, labels=None,
                       bounds=None, hist_kwargs={},
                       filename=None, convert_to_live_points=False):
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
    labels : array_like, optional
        Array of labels for each structured array being plotted (default None).
        If None each set of live points is labelled numerically
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
            live_points[i] = \
                {k: v for k, v in zip(parameters, live_points[i].T)}

    elif any(lp.dtype.names is None for lp in live_points):
        raise RuntimeError('Live points are not structured arrays'
                           'Set `convert_to_live_points=True`.')
    elif parameters is None:
        parameters = live_points[0].dtype.names

    if labels is None:
        labels = [str(i) for i in range(len(live_points))]

    fig, axs = plt.subplots(len(parameters), 1, sharey=False,
                            figsize=(3, 3 * len(parameters)))

    if len(parameters) > 1:
        axs = axs.ravel()
    else:
        axs = [axs]

    for i, f in enumerate(parameters):
        xmin = np.min([np.min(lp[f][np.isfinite(lp[f])])
                       for lp in live_points])
        xmax = np.max([np.max(lp[f][np.isfinite(lp[f])])
                      for lp in live_points])
        for j, lp in enumerate(live_points):
            axs[i].hist(lp[f][np.isfinite(lp[f])],
                        bins=auto_bins(lp[f][np.isfinite(lp[f])]),
                        histtype='step',
                        range=(xmin, xmax), density=True, label=labels[j],
                        **hist_kwargs)
        axs[i].set_xlabel(f)
        if bounds is not None and f in bounds:
            axs[i].axvline(bounds[f][0], ls=':', alpha=0.5, color='k')
            axs[i].axvline(bounds[f][1], ls=':', alpha=0.5, color='k')

    if len(labels) > 1:
        handles, labels = plt.gca().get_legend_handles_labels()
        legend_labels = dict(zip(labels, handles))
        fig.legend(legend_labels.values(), legend_labels.keys(),
                   frameon=False, ncol=len(labels),
                   loc='upper center',
                   bbox_to_anchor=(0, 0.1 / len(parameters), 1, 1),
                   bbox_transform=plt.gcf().transFigure)

    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_indices(indices, nlive=None, weights=None, plot_breakdown=True,
                 filename=None):
    """
    Histogram indices for index insertion tests, also includes the CDF.

    Parameters
    ----------
    indices : array_like
        List of insertion indices to plot
    nlive : int
        Number of live points used in the nested sampling run
    weights : array_like, optional
        Weights for each sample.
    plot_breakdown : bool, optional
       If true, then the CDF for every nlive points is also plotted as grey
       lines.
    filename : str
        Filename used to save the figure.
    """
    indices = np.asarray(indices)
    if not indices.size or not nlive:
        logger.warning('Not producing indices plot.')
        return
    if weights is not None:
        weights = np.asarray(weights)
        print(weights)
        if not indices.size == weights.size:
            raise ValueError(
                'Indices and weights arrays are different sizes: '
                f'{indices.size}, {weights.size}.'
            )
    else:
        weights = np.ones(len(indices))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    nbins = min(len(np.histogram_bin_edges(indices, 'auto')) - 1, 1000)
    if plot_breakdown:
        for i in range(len(indices) // nlive):
            ax[1].hist(indices[i * nlive:(i+1) * nlive], bins=nlive,
                       histtype='step', density=True, alpha=0.1, color='black',
                       lw=0.5, cumulative=True, range=(0, nlive-1),
                       weights=weights[i * nlive:(i+1) * nlive])

    ax[0].hist(indices, density=True, color='tab:blue', linewidth=1.25,
               histtype='step', bins=nbins, label='produced',
               range=(0, nlive-1), weights=weights)
    ax[1].hist(indices, density=True, color='tab:blue', linewidth=1.25,
               histtype='step', bins=nlive, label='produced',
               cumulative=True, range=(0, nlive-1), weights=weights)

    if nlive is not None:
        ax[0].axhline(1 / nlive, color='black', linewidth=1.25,
                      linestyle='-', label='pmf', alpha=0.5)
        ax[0].axhline((1 + (nbins / len(indices)) ** 0.5) / nlive,
                      color='black', linewidth=1.25, linestyle=':', alpha=0.5,
                      label='1-sigma')
        ax[0].axhline((1 - (nbins / len(indices)) ** 0.5) / nlive,
                      color='black', linewidth=1.25, linestyle=':', alpha=0.5)
        ax[1].plot([0, nlive], [0, 1], color='black', linewidth=1.25,
                   linestyle=':', label='cmf')

    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right')
    ax[0].set_xlim([0, nlive-1])
    ax[1].set_xlim([0, nlive-1])
    ax[0].set_xlabel('Insertion indices')
    ax[1].set_xlabel('Insertion indices')

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


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
    plt.plot(epochs, history['loss'], label='Loss')
    plt.plot(epochs, history['val_loss'], label='Val. loss')
    plt.xlabel('Epochs')
    plt.ylabel('Negative log-likelihood')
    plt.legend()
    plt.tight_layout()
    if any(h < 0 for h in history['loss']):
        plt.yscale('symlog')
    else:
        plt.yscale('log')

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_trace(log_x, nested_samples, labels=None, filename=None):
    """
    Produce trace plot for all of the parameters.

    Parameters
    ----------
    log_x : array_like
        Array of log prior volumnes
    nested_samples : ndrray
        Array of nested samples to plot
    labels : list, optional
        List of labels to use instead of the names of parameters
    filename : str, optional
        Filename for saving the plot, if none plot is not saved and figure
        is returned instead.
    """
    nested_samples = np.asarray(nested_samples)
    if not nested_samples.dtype.names:
        raise TypeError('Nested samples must be a structured array')

    names = [n for n in nested_samples.dtype.names
             if n not in ['logP', 'logW', 'logL']]

    if labels is None:
        labels = names

    if not len(labels) == len(names):
        raise RuntimeError(
            'Missing labels. List of labels does not have enough entries '
            f'({len(labels)}) for parameters: {nested_samples.dtype.names}')

    fig, axes = plt.subplots(len(labels), 1, figsize=(5, 3 * len(labels)),
                             sharex=True)
    if len(labels) > 1:
        axes = axes.ravel()
    else:
        axes = [axes]

    for i, name in enumerate(names):
        axes[i].plot(log_x, nested_samples[name], ',')
        axes[i].set_ylabel(labels[i])

    axes[-1].set_xlabel('log X')
    axes[-1].invert_xaxis()

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_histogram(samples, label=None, filename=None, **kwargs):
    """Plot a histgram of samples.

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
    default_kwargs = dict(
        histtype='step',
        bins=auto_bins(samples)
    )
    default_kwargs.update(kwargs)
    fig = plt.figure()
    plt.hist(samples, **default_kwargs)
    if label is not None:
        plt.xlabel(label)
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig
