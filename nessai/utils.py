import json
import logging
import os
import shutil

from nflows.distributions.uniform import BoxUniform
import numpy as np
from scipy import stats, spatial
import torch
from torch.distributions import MultivariateNormal

from .livepoint import live_points_to_dict

logger = logging.getLogger(__name__)


def logit(x, fuzz=0):
    """
    Logit function that also returns log Jacobian

    Parameters
    ----------
    x : array_like
        Array of values
    fuzz : float, optional
        Fuzz used to avoid nans in logit. Values are rescaled from [0, 1]
        to [0-fuzz, 1+fuzz]. By default no fuzz is applied
    """
    x += fuzz
    x /= (1 + 2 * fuzz)
    return np.log(x) - np.log(1 - x), -np.log(np.abs(x - x ** 2))


def sigmoid(x, fuzz=0):
    """
    Sigmoid function that also returns log Jacobian

    Parameters
    ----------
    x : array_like
        Array of values
    fuzz : float, optional
        Fuzz used to avoid nans in logit
    """
    x = np.asarray(x)
    x = np.divide(1, 1 + np.exp(-x))
    log_J = np.log(np.abs(x - x ** 2))
    x *= (1 + 2 * fuzz)
    x -= fuzz
    return x, log_J


rescaling_functions = {'logit': (logit, sigmoid)}


def compute_indices_ks_test(indices, nlive, mode='D+'):
    """
    Compute the two-sided KS test for discrete insertion indices for a given
    number of live points

    Parameters
    ----------
    indices: array_like
        Indices of newly inserteed live points
    nlive: int
        Number of live points

    Returns
    ------
    D: float
        Two-sided KS statistic
    p: float
        p-value
    """
    if len(indices):
        counts = np.zeros(nlive)
        u, c = np.unique(indices, return_counts=True)
        counts[u] = c
        cdf = np.cumsum(counts) / len(indices)
        if mode == 'D+':
            D = np.max(np.arange(1.0, nlive + 1) / nlive - cdf)
        elif mode == 'D-':
            D = np.max(cdf - np.arange(0.0, nlive) / nlive)
        else:
            raise RuntimeError(f'{mode} is not a valid mode. Choose D+ or D-')
        p = stats.ksone.sf(D, len(indices))
        return D, p
    else:
        return None, None


def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply the Bonferroni correction for multiple tests.

    Based on the implementation in `statmodels.stats.multitest`

    Parameters
    ----------
    p_values :  array_like, 1-d
        Uncorrelated p-values
    alpha : float, optional
        Family wise error rate
    """
    p_values = np.asarray(p_values)
    alpha_bon = alpha / p_values.size
    reject = p_values <= alpha_bon
    p_values_corrected = p_values * p_values.size
    p_values_corrected[p_values_corrected > 1] = 1
    return reject, p_values_corrected, alpha_bon


def draw_surface_nsphere(dims, r=1, N=1000):
    """
    Draw N points uniformly from  n-1 sphere of radius r using Marsaglia's
    algorithm. E.g for 3 dimensions returns points on a 'regular' sphere.

    See Marsaglia (1972)

    Parameters
    ----------
    dims : int
        Dimension of the n-sphere
    r : float, optional
        Radius of the n-sphere, if specified it is used to rescale the samples
    N : int, optional
        Number of samples to draw

    Returns
    -------
    array_like
        Array of samples with shape (N, dims)
    """
    x = np.random.randn(N, dims)
    R = np.sqrt(np.sum(x ** 2., axis=1))[:, np.newaxis]
    z = x / R
    return r * z


def draw_nsphere(dims, r=1, N=1000, fuzz=1.0):
    """
    Draw N points uniformly within an n-sphere of radius r

    Parameters
    ----------
    dims : int
        Dimension of the n-sphere
    r : float, optional
        Radius of the n-ball
    N : int, optional
        Number of samples to draw
    fuzz : float, optional
        Fuzz factor by which to increase the radius of the n-ball

    Returns
    -------
    array_like
        Array of samples with shape (N, dims)
    """
    x = draw_surface_nsphere(dims, r=1, N=N)
    R = np.random.uniform(0, 1, (N, 1))
    z = R ** (1 / dims) * x
    return fuzz * r * z


def get_uniform_distribution(dims, r, device='cpu'):
    """
    Return a Pytorch distribution that is uniform in the number of
    dims specified

    Parameters
    ----------
    dims: int
        Number of dimensions
    r: float
        Radius to use for lower and upper bounds
    device: str, optional (cpu)
        Device on which the distribution is placed.

    Returns
    -------
    :obj:`nflows.distributions.uniform.BoxUniform`
        Instance of BoxUniform which the lower and upper bounds set by
        the radius
    """
    r = r * torch.ones(dims, device=device)
    return BoxUniform(low=-r, high=r)


def get_multivariate_normal(dims, var=1, device='cpu'):
    """
    Return a Pytorch distribution that is normally distributed in n dims
    with a given variance.

    Parameters
    ----------
    dims: int
        Number of dimensions
    var: float, optional (1)
        Standard deviation
    device: str, optional (cpu)
        Device on which the distribution is placed.

    Returns
    -------
        Instance of MultivariateNormal with correct variance and dims
    """
    loc = torch.zeros(dims).to(device).double()
    covar = var * torch.eye(dims).to(device).double()
    return MultivariateNormal(loc, covariance_matrix=covar)


def draw_uniform(dims, r=(1,), N=1000, fuzz=1.0):
    """
    Draw from a uniform distribution on [0, 1], deals with extra input
    parameters used by other draw functions
    """
    return np.random.uniform(0, 1, (N, dims))


def draw_gaussian(dims, r=1, N=1000, fuzz=1.0):
    """
    Wrapper for numpy.random.randn that deals with extra input parameters
    r and fuzz

    Parameters
    ----------
    dims : int
        Dimension of the n-sphere
    r : float, optional
        Radius of the n-ball
    N : int, ignored
        Number of samples to draw
    fuzz : float, ignored
        Fuzz factor by which to increase the radius of the n-ball

    Returns
    -------
    array_like
        Array of samples with shape (N, dims)
    """
    return np.random.randn(N, dims)


def draw_truncated_gaussian(dims, r, N=1000, fuzz=1.0, var=1):
    """
    Draw N points from a truncated gaussian with a given a radius

    Parameters
    ----------
    dims : int
        Dimension of the n-sphere
    r : float
        Radius of the truncated Gaussian
    N : int, ignored
        Number of samples to draw
    fuzz : float, ignored
        Fuzz factor by which to increase the radius of the truncated Gaussian

    Returns
    -------
    array_like
        Array of samples with shape (N, dims)
    """
    sigma = np.sqrt(var)
    r *= fuzz
    u_max = stats.chi.cdf(r / sigma, df=dims)
    u = np.random.uniform(0, u_max, N)
    p = sigma * stats.chi.ppf(u, df=dims)
    x = np.random.randn(p.size, dims)
    points = (p * x.T / np.sqrt(np.sum(x**2., axis=1))).T
    return points


def replace_in_list(target_list, targets, replacements):
    """
    Replace (in place) an entry in a list with a given element

    Parameters
    ----------
    target_list : list
        List to update
    targets : list
        List of items to update
    replacements : list
        List of replacement items
    """
    if not isinstance(targets, list):
        if isinstance(targets, int):
            targets = [targets]
        else:
            targets = list(targets)
    if not isinstance(replacements, list):
        if isinstance(replacements, int):
            replacements = [replacements]
        else:
            replacements = list(replacements)

    if not all([t in target_list for t in targets]):
        raise ValueError(f'Targets {targets} not in list: {target_list}')

    for t, r in zip(targets, replacements):
        i = target_list.index(t)
        target_list[i] = r


def rescale_zero_to_one(x, xmin, xmax):
    """
    Rescale a value to 0 to 1

    Parameters
    ----------
    x : array_like
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    array_like
        Array of rescaled values
    array_like
        Array of log determinants of Jacobians for each sample
    """
    return (x - xmin) / (xmax - xmin), -np.log(xmax - xmin)


def inverse_rescale_zero_to_one(x, xmin, xmax):
    """
    Rescale from 0 to 1 to xmin to xmax

    Parameters
    ----------
    x : array_like
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    array_like
        Array of rescaled values
    array_like
        Array of log determinants of Jacobians for each sample
    """
    return (xmax - xmin) * x + xmin, np.log(xmax - xmin)


def rescale_minus_one_to_one(x, xmin, xmax):
    """
    Rescale a value to -1 to 1

    Parameters
    ----------
    x : array_like
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    array_like
        Array of rescaled values
    array_like
        Array of log determinants of Jacobians for each sample
    """
    return ((2. * (x - xmin) / (xmax - xmin)) - 1,
            np.log(2) - np.log(xmax - xmin))


def inverse_rescale_minus_one_to_one(x, xmin, xmax):
    """
    Rescale from -1 to 1 to xmin to xmax

    Parameters
    ----------
    x : array_like
        Array of values to rescale
    xmin, xmax : floats
        Minimum and maximum values to use for rescaling

    Returns
    -------
    array_like
        Array of rescaled values
    array_like
        Array of log determinants of Jacobians for each sample
    """
    return ((xmax - xmin) * ((x + 1) / 2.) + xmin,
            np.log(xmax - xmin) - np.log(2))


def detect_edge(x, x_range=None, percent=0.1, cutoff=0.5, nbins='auto',
                allow_both=False, allow_none=False,
                allowed_bounds=['lower', 'upper'], test=None):
    """
    Detect edges in input distributions based on the density.

    Parameters
    ----------
    x: array_like
        Samples
    x_range : array_like, optional
        Lower and upper bounds used to check inversion, if not specified
        min and max of data are used.
    percent: float (0.1)
        Percentage of interval used to check edges
    cutoff: float (0.1)
        Minimum fraction of the maximum density contained within the
        percentage of the interval specified
    allow_both: bool
        Allow function to return both instead of force either upper or lower
    allow_none: bool
        Allow for neither lower or upper bound to be returned
    test : str or None
        If not None this skips the process and just returns the value of test.
        This is used to verify the inversion in all possible scenarios.

    Returns
    -------
    str or False, {'lower', 'upper', 'both', False}
        Returns the boundary to apply the inversion or False is no inversion
        is to be applied
    """
    bounds = ['lower', 'upper']
    if test is not None:
        if test in bounds and test not in allowed_bounds:
            logger.debug(f'{test} is not an allowed bound, returning False')
            return False
        else:
            return test
    if not all(b in bounds for b in allowed_bounds):
        raise RuntimeError(f'Unknown allowed bounds: {allowed_bounds}')
    if nbins == 'auto':
        nbins = auto_bins(x)

    hist, bins = np.histogram(x, bins=nbins, density=True, range=x_range)
    n = max(int(len(bins) * percent), 1)
    bounds_fraction = \
        np.array([np.sum(hist[:n]), np.sum(hist[-n:])]) * (bins[1] - bins[0])
    max_idx = np.argmax(hist)
    max_density = hist[max_idx] * (bins[1] - bins[0])
    logger.debug(f'Max. density: {max_density:.3f}')

    for i, b in enumerate(bounds):
        if b not in allowed_bounds:
            bounds.pop(i)
            bounds_fraction = np.delete(bounds_fraction, i)
    if max_idx <= n and 'lower' in bounds:
        return bounds[0]
    elif max_idx >= (len(bins) - n) and 'upper' in bounds:
        return bounds[-1]
    elif not np.any(bounds_fraction > cutoff * max_density) and allow_none:
        logger.debug('Density too low at both bounds')
        return False
    else:
        if (np.all(bounds_fraction > cutoff * max_density) and allow_both and
                len(bounds) > 1):
            logger.debug('Both bounds above cutoff')
            return 'both'
        else:
            return bounds[np.argmax(bounds_fraction)]


def configure_edge_detection(d, detect_edges):
    """
    Configure parameters for edge detection

    Parameters
    ----------
    d : dict
        Dictionary of kwargs parsed to detect_edge
    detect_edges : bool
        If true allows for no inversion to be applied

    Returns
    -------
    dict
        Updated kwargs
    """
    default = dict(cutoff=0.5)
    if d is None:
        d = {}
    if detect_edges:
        d['allow_none'] = True
    else:
        d['allow_none'] = False
        d['cutoff'] = 0.0
    default.update(d)
    logger.debug(f'detect edges kwargs: {default}')
    return default


def compute_minimum_distances(samples, metric='euclidean'):
    """
    Compute the distance to the nearest neighbour of each sample

    Parameters
    ----------
    samples : array_like
        Array of samples
    metric : str, optional (euclidean)
        Metric to use. See scipy docs for list of metrics:
        https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.spatial.distance.cdist.html

    Returns
    -------
    array_like
        Distance to nearest neighbour for each sample
    """
    d = spatial.distance.cdist(samples, samples, metric)
    d[d == 0] = np.nan
    dmin = np.nanmin(d, axis=1)
    return dmin


def setup_logger(output=None, label='nessai', log_level='INFO'):
    """
    Setup logger

    Based on the implementation in Bilby

    Parameters
    ----------
    output : str, optional
        Path of to output directory
    label : str, optional
        Label for this instance of the logger
    log_level : {'ERROR', 'WARNING', 'INFO', 'DEBUG'}
        Level of logging parsed to logger

    Returns
    -------
    logger
    """
    from . import __version__ as version
    if type(log_level) is str:
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError('log_level {} not understood'.format(log_level))
    else:
        level = int(log_level)

    logger = logging.getLogger('nessai')
    logger.propagate = False
    logger.setLevel(level)

    if any([type(h) == logging.StreamHandler for h in logger.handlers]) \
            is False:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s',
            datefmt='%m-%d %H:%M'))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
        if label:
            if output:
                if not os.path.exists(output):
                    os.makedirs(output, exist_ok=True)
            else:
                output = '.'
            log_file = '{}/{}.log'.format(output, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    logger.info(f'Running Nessai version {version}')

    return logger


def is_jsonable(x):
    """
    Check if an object is JSON serialisable

    Based on: https://stackoverflow.com/a/53112659
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class FPJSONEncoder(json.JSONEncoder):
    """
    Class to encode numpy arrays and other non-serialisable objects in
    FlowProposal

    Based on: https://stackoverflow.com/a/57915246
    """
    def default(self, obj):

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif not is_jsonable(obj):
            return str(obj)
        else:
            return super(FPJSONEncoder, self).default(obj)


def safe_file_dump(data, filename, module, save_existing=False):
    """ Safely dump data to a .pickle file

    See Bilby for the original impletmentation:
    https://git.ligo.org/michael.williams/bilby/-/blob/master/bilby/core/utils.py

    Parameters
    ----------
    data:
        data to dump
    filename: str
        The file to dump to
    module: pickle, dill
        The python module to use
    """
    if save_existing:
        if os.path.exists(filename):
            old_filename = filename + ".old"
            shutil.move(filename, old_filename)

    temp_filename = filename + ".temp"
    with open(temp_filename, "wb") as file:
        module.dump(data, file)
    shutil.move(temp_filename, filename)


def save_live_points(live_points, filename):
    """
    Save live points to a file. Live points are converted to a dictionary
    and then saved.
    """
    d = live_points_to_dict(live_points)
    with open(filename, 'w') as wf:
        json.dump(d, wf, indent=4, cls=FPJSONEncoder)


def configure_threads(max_threads=None, pytorch_threads=None, n_pool=None):
    """
    Configure the number of threads available. This is necessary when using
    PyTorch on the CPU as by default it will use all available threads.

    Notes
    -----
    Uses torch.set_num_threads. If pytorch threads is None but other
    arguments are specified them the value is inferred from them.

    Parameters
    ----------
    max_threads: int (None)
        Maximum total number of threads to use between PyTorch and
        multiprocessing
    pytorch_threads: int (None)
        Maximum number of threads for PyTorch on CPU
    n_pool: int (None)
        Number of pools to use if using multiprocessing
    """
    if max_threads is not None:
        if pytorch_threads is not None and pytorch_threads > max_threads:
            raise RuntimeError(
                f'More threads assigned to PyTorch ({pytorch_threads}) '
                f'than are available ({max_threads})')
        if n_pool is not None and n_pool >= max_threads:
            raise RuntimeError(
                f'More threads assigned to pool ({n_pool}) than are '
                f'available ({max_threads})')
        if (n_pool is not None and pytorch_threads is not None and
                (pytorch_threads + n_pool) > max_threads):
            raise RuntimeError(
                f'More threads assigned to PyTorch ({pytorch_threads}) '
                f'and pool ({n_pool})than are available ({max_threads})')

    if pytorch_threads is None:
        if max_threads is not None:
            if n_pool is not None:
                pytorch_threads = max_threads - n_pool
            else:
                pytorch_threads = max_threads

    if pytorch_threads is not None:
        logger.debug(
            f'Setting maximum number of PyTorch threads to {pytorch_threads}')
        torch.set_num_threads(pytorch_threads)


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


def auto_bins(x, max_bins=50):
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


def determine_rescaled_bounds(prior_min, prior_max, x_min, x_max, invert,
                              offset=0):
    """
    Determine the values of the prior min and max in the rescaled
    space.

    Parameters
    ----------
    prior_min : float
        Mininum of the prior
    prior_max : float
        Maximum of the prior
    x_min : float
        New minimum
    x_max : float
        New maximum
    invert : false or {'upper', 'lower', 'both'}
        Type of inversion
    """
    lower = (prior_min - offset - x_min) / (x_max - x_min)
    upper = (prior_max - offset - x_min) / (x_max - x_min)
    if not invert or invert is None:
        return 2 * lower - 1, 2 * upper - 1
    elif invert == 'upper':
        return lower - 1, 1 - lower
    elif invert == 'lower':
        return -upper, upper
    elif invert == 'both':
        return -0.5, 1.5
    else:
        raise RuntimeError
