# -*- coding: utf-8 -*-
"""
Functions related to computing the evidence.
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
    """
    Object to handle computing the evidence for importance nested sampling.

    Parameters
    ----------
    normalised
        Indicates if the samples being added are correctly normalised. That is,
        computing logsumexp(logL + logW) is normalised by default. If not, then
        this should false and the normalisation constant for logQ should set
        using :code:`log_meta_constant`.
    """

    def __init__(self, normalised: bool = True) -> None:
        self._n = 0
        self._logZ = -np.inf
        self._logZ_history = np.empty(0)
        # Constant to normalise the meta proposal
        self.normalised = normalised
        self._log_meta_constant = None

    def update_evidence(self, x: np.ndarray) -> None:
        """Update the evidence estimate with new samples (live points)"""
        log_Z_k = x['logL'] + x['logW']
        self._logZ_history = np.concatenate([self._logZ_history, log_Z_k])
        self._logZ = logsumexp(self._logZ_history)
        self._n += x.size

    def update_evidence_from_nested_samples(self, x: np.ndarray) -> None:
        """Update the evidence from a set of nested samples"""
        log_Z_k = x['logL'] + x['logW']
        self._logZ_history = log_Z_k
        self._logZ = logsumexp(log_Z_k)
        self._n = x.size

    @property
    def renormalise(self) -> bool:
        """Indicates where the evidence needs renormalising"""
        # Value can only be set if normalised=False
        if self._log_meta_constant is not None:
            return True
        elif not self.normalised:
            raise RuntimeError(
                '`normalised=False` but the constant is not set'
            )
        else:
            return False

    @property
    def log_meta_constant(self) -> float:
        """Constant to normalise the meta proposal weights.

        This should be the log of the value by which Q should be divided to
        normalise it. For example, in the case the normlisation should be
        Q/N, then the constant should log(N).

        Should be set when the weights used to compute logW are not
        normalised and are normalised using this constant.
        """
        if self._log_meta_constant is None:
            if not self.normalised:
                raise RuntimeError(
                    'Samples are not correctly normalised and require '
                    'renormalising but the constant has not been set!'
                )
            return np.log(self._n)
        else:
            return self._log_meta_constant

    @log_meta_constant.setter
    def log_meta_constant(self, value) -> None:
        if self.normalised:
            raise RuntimeError(
                'Weights are normalised. Cannot set the meta constant!'
            )
        self._log_meta_constant = value

    @property
    def log_constant(self) -> float:
        """Constant to renormalise the evidence"""
        if self.renormalise:
            return self._log_meta_constant - np.log(self._n)
        else:
            return 0.0

    @property
    def logZ(self) -> float:
        """The current log-evidence."""
        return self._logZ + self.log_constant

    log_evidence = logZ
    """Alias for logZ"""

    @property
    def log_evidence_error(self) -> float:
        """Alias for compute_uncertainty"""
        return self.compute_uncertainty()

    def compute_updated_log_Z(self, samples: np.ndarray) -> float:
        """Compute the evidence if a set of samples were added.

        Does not update the running estimate of log Z.
        """
        log_Z_s = logsumexp(samples['logL'] + samples['logW'])
        logZ = np.logaddexp(self._logZ, log_Z_s)
        if self.renormalise:
            logZ += (self._log_meta_constant - np.log(self._n + samples.size))
        return logZ

    def compute_condition(self, samples: np.ndarray) -> float:
        """Compute the fraction change in the evidence.

        If samples is None or empty, returns zero.
        """
        if samples is None or not len(samples):
            return 0.0
        logZ = self.compute_updated_log_Z(samples)
        logger.debug(f'Current log Z: {self.logZ}, expected: {logZ}')
        dZ = logZ - self.logZ
        return dZ

    def compute_uncertainty(self) -> float:
        """Compute the uncertainty on the current estimate of the evidence."""
        n = self._n
        Z_hat = np.exp(self.logZ, dtype=np.float128)
        # Need to include a constant to correctly normalise the meta proposal
        # this should be log(N) if the weights are equal to the number of
        # samples, but can be different if the weights were set differently.
        Z = np.exp(
            self._logZ_history + self.log_meta_constant,
            dtype=np.float128,
        )
        # Standard error sqrt(Var[Z] / n)
        u = np.sqrt(np.sum((Z - Z_hat) ** 2) / (n * (n - 1)))
        # sigma[ln Z] = |sigma[Z] / Z|
        return float(np.abs(u / Z_hat))

    @property
    def log_posterior_weights(self) -> np.ndarray:
        """Compute the weights for all of the dead points."""
        return self._logZ_history + self.log_meta_constant - self.logZ

    @property
    def effective_n_posterior_samples(self) -> float:
        """Kish's effective sample size"""
        log_p = self.log_posterior_weights
        if not len(log_p):
            return 0
        log_p -= logsumexp(log_p)
        n = np.exp(-logsumexp(2 * log_p))
        return n
