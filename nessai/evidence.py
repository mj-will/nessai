# -*- coding: utf-8 -*-
"""
Functions related to computing the evidence.
"""
import logging
from typing import Optional

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
        self._n_ns = 0
        self._n_lp = 0
        self._logZ = -np.inf
        self._weights_ns = None
        self._weights_lp = None
        self._weights = None
        # Constant to normalise the meta proposal
        self.normalised = normalised
        self._log_meta_constant = None

    def update_evidence(
        self,
        nested_samples: np.ndarray,
        live_points: Optional[np.ndarray] = None,
    ) -> None:
        """Update the evidence.

        Parameters
        ----------
        nested_samples
            Array of nested samples.
        live_points
            Optional array of live points, if included the evidence will
            include both live points and nested samples. If not, the evidence
            will only include the nested samples.
        """
        self._weights_ns = nested_samples['logL'] + nested_samples['logW']
        if live_points is not None:
            self._weights_lp = live_points['logL'] + live_points['logW']
            self._weights = np.concatenate([
                self._weights_ns, self._weights_lp,
            ])
        else:
            self._weights = self._weights_ns
            self._weights_lp = None
        self._logZ = logsumexp(self._weights)
        self._n = self._weights.size

    @property
    def renormalise(self) -> bool:
        """Indicates where the evidence needs renormalising"""
        if not self.normalised:
            if self._log_meta_constant is not None:
                return True
            else:
                raise RuntimeError(
                    '`normalised=False` but the constant is not set'
                )
        else:
            return False

    @property
    def log_meta_constant(self) -> float:
        """Constant to normalise the meta proposal weights.

        This should be the log of the value by which Q should be divided to
        normalise it. For example, in the case the normalisation should be
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
        self._log_meta_constant = value

    @property
    def log_constant(self) -> float:
        """Constant to renormalise the evidence.

        If :code:`renormalise` is :code:`False` returns zero.
        """
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

    @property
    def log_evidence_live_points(self) -> float:
        """Log-evidence in the live points

        Requires that the log-meta constant is set.
        """
        if self._weights_lp is None:
            raise RuntimeError('Live points are not set')
        return (
            logsumexp(self._weights_lp)
            + self._log_meta_constant
            - np.log(self._weights_lp.size)
        )

    @property
    def log_evidence_nested_samples(self):
        """Log-evidence in the nested samples

        Requires that the log-meta constant is set.
        """
        return (
            logsumexp(self._weights_ns)
            + self._log_meta_constant
            - np.log(self._weights_ns.size)
        )

    def compute_evidence_ratio(self, ns_only: bool = False) -> float:
        """
        Compute the ratio of the evidence in the live points to the nested
        samples.

        Parameters
        ----------
        ns_only
            If True only the evidence from the nested samples is used in the
            denominator of the ratio.

        Returns
        -------
        float
            Log ratio of the evidence
        """
        if ns_only:
            return (
                self.log_evidence_live_points
                - self.log_evidence_nested_samples
            )
        else:
            return self.log_evidence_live_points - self.logZ

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
            self._weights + self.log_meta_constant,
            dtype=np.float128,
        )
        # Standard error sqrt(Var[Z] / n)
        u = np.sqrt(np.sum((Z - Z_hat) ** 2) / (n * (n - 1)))
        # sigma[ln Z] = |sigma[Z] / Z|
        return float(np.abs(u / Z_hat))

    @property
    def log_posterior_weights(self) -> np.ndarray:
        """Compute the log posterior weights.

        If the live points have been specified, then weights will be computed
        for these as well.
        """
        return self._weights + self.log_meta_constant - self.logZ

    @property
    def effective_n_posterior_samples(self) -> float:
        """Kish's effective sample size.

        If the live points have been specified, then their weights will be
        included when computing the ESS.

        Returns
        -------
        float
            The effective samples size. Returns zero if the posterior weights
            are empty.
        """
        log_p = self.log_posterior_weights
        if not len(log_p):
            return 0
        log_p -= logsumexp(log_p)
        n = np.exp(-logsumexp(2 * log_p))
        return n
