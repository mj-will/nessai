# -*- coding: utf-8 -*-
"""
Functions realted to computing the evidence.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


def logsubexp(x, y):
    """
    Helper function to compute the exponential
    of a difference between two numbers

    Computes: ``x + np.log1p(-np.exp(y-x))``

    Parameters
    ----------
    x, y : float or array_like
        Inputs
    """
    if np.any(x < y):
        raise RuntimeError('cannot take log of negative number '
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
    def __init__(self, nlive, track_gradients=True):
        self.nlive = nlive
        self.reset()
        self.track_gradients = track_gradients

    def reset(self):
        """
        Reset the sampler to its initial state at logZ = -infinity
        """
        self.logZ = -np.inf
        self.oldZ = -np.inf
        self.logw = 0
        self.info = [0.]
        # Start with a dummy sample enclosing the whole prior
        self.logLs = [-np.inf]   # Likelihoods sampled
        self.log_vols = [0.0]    # Volumes enclosed by contours
        self.gradients = [0]

    def increment(self, logL, nlive=None):
        """
        Increment the state of the evidence integrator
        Simply uses rectangle rule for initial estimate
        """
        if (logL <= self.logLs[-1]):
            logger.warning('NS integrator received non-monotonic logL.'
                           f'{self.logLs[-1]:.5f} -> {logL:.5f}')
        if nlive is None:
            nlive = self.nlive
        oldZ = self.logZ
        logt = - 1.0 / nlive
        Wt = self.logw + logL + np.log1p(-np.exp(logt))
        self.logZ = np.logaddexp(self.logZ, Wt)
        # Update information estimate
        if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
            info = np.exp(Wt - self.logZ) * logL \
                  + np.exp(oldZ - self.logZ) \
                  * (self.info[-1] + oldZ) \
                  - self.logZ
            if np.isnan(info):
                info = 0
            self.info.append(info)

        # Update history
        self.logw += logt
        self.logLs.append(logL)
        self.log_vols.append(self.logw)
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

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
            plt.close()
            logger.info(f'Saved nested sampling plot as {filename}')
        else:
            return fig


class _INSIntegralState:

    def __init__(self, **kwargs):
        self._n = 0
        self._logZ = -np.inf
        self.info = [np.nan]
        self.log_vols = [0.]
        self.logX = 0.
        self.effective_sample_size = 0
        self._logZ_history = np.empty(0)

    def update_evidence(self, x: np.ndarray):
        """Update the evidence estimate."""
        log_Z_k = x['logL'] + x['logW']
        self._logZ_history = np.concatenate([self._logZ_history, log_Z_k])
        self._logZ = logsumexp(self._logZ_history)
        self._n += x.size

    def update_evidence_from_nested_samples(self, x):
        log_Z_k = x['logL'] + x['logW']
        self._logZ_history = log_Z_k
        self._logZ = logsumexp(log_Z_k)
        self._n = x.size

    @property
    def logZ(self):
        return self._logZ

    def finalise(self):
        logger.debug('Calling finalise. Nothing happened!')
        pass

    def compute_condition(self, logL, logW) -> float:
        logZ = np.logaddexp(self._logZ, logL + logW)
        dZ = logZ - self.logZ
        return dZ

    def compute_uncertainty(self) -> float:
        n = self._n
        Z_hat = np.exp(self.logZ, dtype=np.float128)
        # Include n since g does not include it
        Z = n * np.exp(self._logZ_history.astype(np.float128))
        # Var[Z]
        u = np.sqrt(np.sum((Z - Z_hat) ** 2) / (n * (n - 1)))
        # sigma[ln Z] = |sigma[Z] / Z|
        return float(np.abs(u / Z_hat))

    @property
    def log_posterior_weights(self):
        """Compute the weights for all of the dead points."""
        return np.asarray(self._logZ_history).copy() - self.logZ

    @property
    def effective_n_posterior_samples(self) -> float:
        """Kish's effectice sample size"""
        log_p = self.log_posterior_weights
        log_p -= logsumexp(log_p)
        n = np.exp(-logsumexp(2 * log_p))
        return n
