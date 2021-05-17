# -*- coding: utf-8 -*-
"""
Functions realted to computing the evidence.
"""
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


class LogNegativeError(ValueError):
    """Error raised when try to compute the log of a negative number"""
    pass


def logsubexp(x, y):
    """
    Helper function to compute the exponential
    of a difference between two numbers

    Computes: ``x + np.log1p(-np.exp(y-x))``

    Parameters
    ----------
    x, y : float or array_like
        Inputs

    Raises
    ------
    LogNegativeError
        If the value of x is less than y and the calculation would require
        computing the log of a negative number.
    """
    if np.any(x < y):
        raise LogNegativeError('cannot take log of negative number '
                               f'{str(x)!s} - {str(y)!s}')

    return x + np.log1p(-np.exp(y - x))


def log_integrate_log_trap(log_func, log_support):
    """
    Trapezoidal integration of given log(func). Returns log of the integral.

    Parameters
    ----------
    log_func : array_like
        Log values of the function to integrate over.
    log_support : array_like
        Log prior-volumes for each value.

    Returns
    -------
    float
        Log of the result of the integral.
    """
    log_func_sum = np.logaddexp(log_func[:-1], log_func[1:]) - np.log(2)
    log_dxs = logsubexp(log_support[:-1], log_support[1:])

    return np.logaddexp.reduce(log_func_sum + log_dxs)


class _NSIntegralState:
    """
    Stores the state of the nested sampling integrator

    Parameters
    ----------
    nlive : int
        Number of live points
    track_gradients : bool, optional
        If true the gradient of the change in logL w.r.t logX is saved each
        time `increment` is called.
    """
    def __init__(self, nlive, G=None, track_gradients=True):
        self.nlive = nlive
        self.G = G
        self.reset()
        self.track_gradients = track_gradients

    def reset(self):
        """
        Reset the sampler to its initial state at logZ = -infinity
        """
        self.logZ = -np.inf
        self.oldZ = -np.inf
        self.logX = 0
        self.info = [0.]
        # Start with a dummy sample enclosing the whole prior
        self.logLs = [-np.inf]   # Likelihoods sampled
        self.log_vols = [0.0]    # Volumes enclosed by contours
        self.gradients = [0]
        self.log_w_norm = []

    def increment(self, x, nlive=None, log_w_norm=None):
        """
        Increment the state of the evidence integrator
        Simply uses rectangle rule for initial estimate
        """
        logL = x['logL']
        logW = x['logW']

        if nlive is None:
            nlive = self.nlive

        if log_w_norm is None:
            log_w_norm = np.log(nlive)
            if not logW == 0.:
                raise ValueError(
                    'Weights must be zero when normalisation is None')

        if (logL <= self.logLs[-1]):
            logger.warning('NS integrator received non-monotonic logL.'
                           f'{self.logLs[-1]:.5f} -> {logL:.5f}')
        oldZ = self.logZ
        logt = -np.exp(logW - log_w_norm)
        Wt = self.logX + logL + np.log1p(-np.exp(logt))
        self.logZ = np.logaddexp(self.logZ, Wt)
        # Update information estimate
        if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
            info = (
                np.exp(Wt - self.logZ) * logL
                + np.exp(oldZ - self.logZ) * (self.info[-1] + oldZ)
                - self.logZ
            )
            if np.isnan(info):
                info = 0
            self.info.append(info)
        else:
            self.info.append(0.0)

        # Update history
        # Current X is just t_i * X_(i-1)
        self.logX += logt
        self.logLs.append(logL)
        self.log_vols.append(self.logX)
        self.log_w_norm.append(log_w_norm)
        if self.track_gradients:
            self.gradients.append((self.logLs[-1] - self.logLs[-2])
                                  / (self.log_vols[-1] - self.log_vols[-2]))

    def finalise(self):
        """
        Compute the final evidence with more accurate integrator
        Call at end of sampling run to refine estimate
        """
        # Trapezoidal rule
        self.logZ = log_integrate_log_trap(np.array(self.logLs),
                                           np.array(self.log_vols))
        return self.logZ

    def log_weights(self, trap=False):
        """Compute the weights for all of the dead points."""
        log_l = np.array(self.logLs + [self.logLs[-1]])
        log_vols = np.array(self.log_vols + [-np.inf])
        log_dvols = logsubexp(log_vols[:-1], log_vols[1:])
        if trap:
            log_dvols -= np.log(2)
        return log_l[1:-1] + log_dvols[:-1]

    @property
    def effective_sample_size(self):
        """Computes the Kish effective sample size"""
        log_w = self.log_weights()
        ess = 2 * logsumexp(log_w) - logsumexp(2 * log_w)
        return np.exp(ess)

    def importance_weights(self, G=None):
        """Compute the importance of all of the samples.

        Based on the implementation in dynesty: \
            https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynamicsampler.py
        """
        if G is None:
            G = self.G
        log_wt = self.log_weights()

        # log_z_remain = self.logLs[-1] + self.log_vols[-1]
        log_z_above = np.logaddexp.accumulate(log_wt)
        # log_z_tot = np.logaddexp(log_z_above, log_z_remain)

        # log_z_in = logsubexp(
        #     log_z_tot * np.ones(log_z_above.size), log_z_above)
        log_z_weight = log_z_above - np.array(self.log_w_norm)
        log_z_weight -= logsumexp(log_z_weight)
        z_weight = np.exp(log_z_weight)
        log_z = logsumexp(log_wt)

        p_weight = np.exp(log_wt - log_z)

        weight = (1 - G) * z_weight + G * p_weight

        return weight, z_weight, p_weight

    def plot(self, filename=None):
        """
        Plot the logX vs logL

        Parameters
        ----------
        filename : str, optional
            Filename name for saving the figure. If not specified the figure
            is returned.
        """
        fig = plt.figure()
        plt.plot(self.log_vols, self.logLs)
        plt.title(f'log Z={self.logZ:.2f} '
                  f'H={self.info[-1] * np.log2(np.e):.2f} bits')
        plt.grid(which='both')
        plt.xlabel('log prior-volume')
        plt.ylabel('log-likelihood')
        plt.xlim([self.log_vols[-1], self.log_vols[0]])
        plt.yscale('symlog')

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
            plt.close()
            logger.info(f'Saved nested sampling plot as {filename}')
        else:
            return fig
