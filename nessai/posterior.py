import numpy as np


def logsubexp(x, y):
    """
    Helper function to compute the exponential
    of a difference between two numbers

    ----------
    Parameter:
        x: :float:
        y: :float:
    ----------
    Return
        z: :float: x + np.log1p(-np.exp(y-x))
    """
    if np.any(x < y):
        raise RuntimeError('cannot take log of negative number '
                           f'{str(x)!s} - {str(y)!s}')

    return x + np.log1p(-np.exp(y - x))


def log_integrate_log_trap(log_func, log_support):
    """
    Trapezoidal integration of given log(func)
    Returns log of the integral
    """

    log_func_sum = np.logaddexp(log_func[:-1], log_func[1:]) - np.log(2)
    log_dxs = logsubexp(log_support[:-1], log_support[1:])

    return np.logaddexp.reduce(log_func_sum + log_dxs)


def compute_weights(data, Nlive):
    """Returns log_ev, log_wts for the log-likelihood samples in data,
    assumed to be a result of nested sampling with Nlive live points."""

    start_data = np.concatenate(([float('-inf')], data[:-Nlive]))
    end_data = data[-Nlive:]

    log_wts = np.zeros(data.shape[0])

    log_vols_start = np.cumsum(np.ones(len(start_data) + 1)
                               * np.log1p(-1. / Nlive)) - np.log1p(-1 / Nlive)
    log_vols_end = np.zeros(len(end_data))
    log_vols_end[-1] = np.NINF
    log_vols_end[0] = log_vols_start[-1] + np.log1p(-1.0 / Nlive)
    for i in range(len(end_data) - 1):
        log_vols_end[i+1] = log_vols_end[i] + np.log1p(-1.0 / (Nlive - i))

    log_likes = np.concatenate((start_data, end_data, [end_data[-1]]))

    log_vols = np.concatenate((log_vols_start, log_vols_end))
    log_ev = log_integrate_log_trap(log_likes, log_vols)

    log_dXs = logsubexp(log_vols[:-1], log_vols[1:])
    log_wts = log_likes[1:-1] + log_dXs[:-1]

    log_wts -= log_ev

    return log_ev, log_wts


def draw_posterior_samples(nested_samples, nlive):
    """
    Draw posterior samples given the nested samples and number
    of live points
    """
    log_Z, log_w = compute_weights(nested_samples['logL'], nlive)
    log_w -= np.max(log_w)
    log_u = np.log(np.random.rand(nested_samples.size))
    return nested_samples[log_w > log_u]
