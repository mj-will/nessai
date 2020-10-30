from matplotlib import ticker
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pandas as pd

sns.set()
sns.set_style('ticks')

pairplot_kwargs = dict(corner=True, kind='scatter',
                       diag_kws=dict(histtype='step', bins='auto', lw=1.5,
                                     density=True, color='teal'),
                       plot_kws=dict(s=1.0, edgecolor=None, palette='viridis',
                                     color='teal'))


def plot_live_points(live_points, filename=None, bounds=None, c=None,
                     **kwargs):
    """
    Plot a set of live points
    """
    pairplot_kwargs.update(kwargs)

    df = pd.DataFrame(live_points)
    df = df[np.isfinite(df).all(1)]

    if c is not None:
        hue = df[c]
    else:
        hue = None
    fig = sns.PairGrid(df, corner=True, diag_sharey=False)
    fig.map_diag(plt.hist, **pairplot_kwargs['diag_kws'])
    fig.map_offdiag(sns.scatterplot, hue=hue,
                    **pairplot_kwargs['plot_kws'])

    if bounds is not None:
        for i, v in enumerate(bounds.values()):
            fig.axes[i, i].axvline(v[0], ls=':', alpha=0.5, color='k')
            fig.axes[i, i].axvline(v[1], ls=':', alpha=0.5, color='k')

    if filename is not None:
        fig.savefig(filename)
    plt.close()


def _hist_bin_fd(x):
    """
    The Freedman-Diaconis histogram bin estimator.

    See original Numpy implementation.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.
    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def _hist_bin_sturges(x):
    """
    Sturges histogram bin estimator.

    See original Numpy implementation.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.
    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    return np.ptp(x) / (np.log2(x.size) + 1.0)


def auto_bins(x, max_bins=100):
    """
    Compute the number bins for a histogram using numpy.histogram_bin_edges
    but enforece a maximum number of bins.

    Parameters
    ----------
    array : array_like
        Input data
    bins : int or sequence of scalars or str, optional
        Method for determining number of bins, see numpy documentation
    max_bins : int, optional (1000)
        Maximum number of bins

    Returns
    -------
    int
        Number of bins
    """
    x = np.asarray(x)
    fd_bw = _hist_bin_fd(x)
    sturges_bw = _hist_bin_sturges(x)
    if fd_bw:
        bw = min(fd_bw, sturges_bw)
    else:
        bw = sturges_bw

    if bw:
        n_bins = int(np.ceil(np.ptp(x)) / bw)
    else:
        n_bins = 1

    nbins = min(n_bins, max_bins)
    assert isinstance(nbins, int)
    return nbins


def plot_1d_comparison(*live_points, parameters=None, labels=None,
                       bounds=None, hist_kwargs={},
                       filename=None, convert_to_live_points=False):
    """
    Plot 1d histograms comparing different sets of live points

    Parameters
    ----------
    *live_points:
        Variable length argument list of live points in structured arrays with
        fields. See `parameters` argument.
    parameters: array_like, optional
        Array of parameters (field names) to plot. Default None implies all
        fields are plotted.
    labels: array_like, optional
        Array of labels for each structured array being plotted (default None).
        If None each set of live points is labelled numerically
    bounds: dict, optional
        Dictionary of upper and lowers bounds to plot. Each key must
        match a field and each value must be an interable of length 2 in order
        lower then upper bound. If None (default), no bounds plotted.
    hist_kwargs: dict, optional
        Dictionary of keyword arguments parsed to matplotlib.pyplot.hist.
    filename: str, optional
        Name of file for saving figure. (Default None implies figure is not
        saved).
    """
    if convert_to_live_points:
        live_points = list(live_points)
        if parameters is None:
            parameters = [i for i in range(live_points[0].shape[-1])]
        for i in range(len(live_points)):
            live_points[i] = {k: v for k, v in enumerate(live_points[i].T)}

    elif parameters is None:
        parameters = live_points[0].dtype.names

    if labels is None:
        labels = [str(i) for i in range(len(live_points))]

    fig, axs = plt.subplots(len(parameters), 1, sharey=False,
                            figsize=(3, 3 * len(parameters)))

    axs = axs.ravel()
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
    plt.close()


def plot_posterior(live_points, filename=None, **kwargs):
    """
    Plot a set of live points
    """
    pairplot_kwargs.update(kwargs)

    df = pd.DataFrame(live_points)
    fig = sns.PairGrid(df)
    fig.map_diag(sns.kdeplot)
    fig.map_offdiag(sns.kdeplot, n_levels=6)

    if filename is not None:
        fig.savefig(filename)
    plt.close()


def plot_indices(indices, nlive=None, u=None, name=None, filename=None,
                 plot_breakdown=True):
    """
    Histogram indices for index insertion tests
    """
    if not indices:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    nbins = min(len(np.histogram_bin_edges(indices, 'auto')) - 1, 1000)
    if plot_breakdown:
        for i in range(len(indices) // nlive):
            ax[1].hist(indices[i * nlive:(i+1) * nlive], bins=nlive,
                       histtype='step', density=True, alpha=0.1, color='black',
                       lw=0.5, cumulative=True, range=(0, nlive-1))

    ax[0].hist(indices, density=True, color='tab:blue', linewidth=1.25,
               histtype='step', bins=nbins, label='produced',
               range=(0, nlive-1))
    ax[1].hist(indices, density=True, color='tab:blue', linewidth=1.25,
               histtype='step', bins=nlive, label='produced',
               cumulative=True, range=(0, nlive-1))

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

    if name is not None:
        ax.set_xlabel(name)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_likelihood_evaluations(evaluations, nlive, filename=None):

    its = np.arange(-1, len(evaluations)) * nlive
    evaluations.insert(0, 0)
    fig = plt.figure()
    plt.plot(its, evaluations, '.')
    plt.xlabel('Iteration')
    plt.ylabel('Number of likelihood evaluations')

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')


def plot_chain(x, name=None, filename=None):
    """
    Produce a trace plot
    """
    fig = plt.figure(figsize=(4, 3))
    plt.plot(x, ',')
    plt.grid()
    plt.xlabel('iteration')
    if name is not None:
        plt.ylabel(name)
        if filename is None:
            filename = name + '_chain.png'
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_flows(model, n_inputs, N=1000, inputs=None, cond_inputs=None,
               mode='inverse', output='./'):
    """
    Plot each stage of a series of flows
    """
    if n_inputs > 2:
        raise NotImplementedError('Plotting for higher dimensions not '
                                  'implemented !')

    outputs = []

    if mode == 'direct':
        if inputs is None:
            raise ValueError('Can not sample from parameter space!')
        else:
            inputs = torch.from_numpy(inputs).to(model.device)

        for module in model._modules.values():
            inputs, _ = module(inputs, cond_inputs, mode)
            outputs.append(inputs.detach().cpu().numpy())
    else:
        if inputs is None:
            inputs = torch.randn(N, n_inputs, device=model.device)
            orig_inputs = inputs.detach().cpu().numpy()
        for module in reversed(model._modules.values()):
            inputs, _ = module(inputs, cond_inputs, mode)
            outputs.append(inputs.detach().cpu().numpy())

    n = int(len(outputs) / 2) + 1
    m = 1

    if n > 5:
        m = int(np.ceil(n / 5))
        n = 5

    z = orig_inputs
    pospos = np.where(np.all(z >= 0, axis=1))
    negneg = np.where(np.all(z < 0, axis=1))
    posneg = np.where((z[:, 0] >= 0) & (z[:, 1] < 0))
    negpos = np.where((z[:, 0] < 0) & (z[:, 1] >= 0))

    points = [pospos, negneg, posneg, negpos]
    colours = ['r', 'c', 'g', 'tab:purple']
    colours = plt.cm.Set2(np.linspace(0, 1, 8))

    fig, ax = plt.subplots(m, n, figsize=(n * 3, m * 3))
    ax = ax.ravel()
    for j, c in zip(points, colours):
        ax[0].plot(z[j, 0], z[j, 1], ',', c=c)
        ax[0].set_title('Latent space')
    for i, o in enumerate(outputs[::2]):
        i += 1
        for j, c in zip(points, colours):
            ax[i].plot(o[j, 0], o[j, 1], ',', c=c)
        ax[i].set_title(f'Flow {i}')
    plt.tight_layout()
    fig.savefig(output + 'flows.png')


def plot_loss(epoch, history, output='./'):
    """
    Plot a loss function
    """
    fig, ax = plt.subplots()
    epochs = np.arange(1, epoch + 1, 1)
    plt.plot(epochs, history['loss'], label='Loss')
    plt.plot(epochs, history['val_loss'], label='Val. loss')
    plt.xlabel('Epochs')
    plt.ylabel('Negative log-likelihood')
    plt.legend()
    plt.tight_layout()
    plt.yscale('symlog')
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.savefig(output + 'loss.png')
    plt.close('all')


def plot_acceptance(*acceptance, filename=None, labels=None):

    if labels is None:
        labels = [f'acceptance_{i}' for i in len(acceptance)]

    fig = plt.figure()
    x = np.arange(len(acceptance[0]))
    for a, l in zip(acceptance, labels):
        plt.plot(a, 'o', label=l)
    plt.xticks(x[::2])
    plt.ylabel('Acceptance')
    plt.legend(frameon=False)
    plt.grid()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
