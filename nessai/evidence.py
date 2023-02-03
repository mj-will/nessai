# -*- coding: utf-8 -*-
"""
Functions related to computing the evidence.
"""
from abc import ABC, abstractmethod
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from .plot import nessai_style

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
        raise RuntimeError(
            "cannot take log of negative number " f"{str(x)!s} - {str(y)!s}"
        )

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

    return logsumexp(log_func_sum + log_dxs)


class _BaseNSIntegralState(ABC):
    """Base class for the nested sampling integral."""

    @property
    @abstractmethod
    def log_evidence(self):
        """The current log-evidence."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def log_evidence_error(self):
        """The current error on the log-evidence."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def log_posterior_weights(self):
        """The log-posterior weights."""
        raise NotImplementedError()

    @property
    def effective_n_posterior_samples(self):
        """Kish's effective sample size for the posterior weights.

        Returns
        -------
        float
            The effective sample size. Returns zero if the posterior weights
            are empty.
        """
        log_p = self.log_posterior_weights
        if not len(log_p):
            return 0
        log_p -= logsumexp(log_p)
        n = np.exp(-logsumexp(2 * log_p))
        return n


class _NSIntegralState(_BaseNSIntegralState):
    """
    Stores the state of the nested sampling integrator

    Parameters
    ----------
    nlive : int
        Number of live points
    track_gradients : bool, optional
        If true the gradient of the change in logL w.r.t logX is saved each
        time `increment` is called.
    expectation : str, {logt, t}
        Method used to compute the expectation value for the shrinkage t.
        Choose between log <t> or <log t>. Defaults to <log t>.
    """

    def __init__(self, nlive, track_gradients=True, expectation="logt"):
        self.base_nlive = nlive
        self.track_gradients = track_gradients

        if expectation.lower() not in ["t", "logt"]:
            raise ValueError(
                f"Expectation must be t or logt, got: {expectation}"
            )
        self.expectation = expectation.lower()

        # Initial state of the integral
        self.logZ = -np.inf
        self.oldZ = -np.inf
        self.logw = 0
        self.info = [0.0]
        # Initially contain all the prior volume
        self.logLs = [-np.inf]  # Likelihoods sampled
        self.log_vols = [0.0]  # Volumes enclosed by contours
        self.nlive = []
        self.gradients = [0]

    @property
    def log_evidence(self):
        """The current log-evidence."""
        return self.logZ

    @property
    def log_evidence_error(self):
        """The current error on the log-evidence."""
        return np.sqrt(self.info[-1] / self.base_nlive)

    def increment(self, logL, nlive=None):
        """
        Increment the state of the evidence integrator
        Simply uses rectangle rule for initial estimate
        """
        if logL <= self.logLs[-1]:
            logger.warning(
                "NS integrator received non-monotonic logL."
                f"{self.logLs[-1]:.5f} -> {logL:.5f}"
            )
        if nlive is None:
            nlive = self.base_nlive

        self.nlive.append(nlive)
        oldZ = self.logZ
        if self.expectation == "logt":
            # <logt> approx -1 / N
            logt = -1.0 / nlive
        else:
            # <t> = N / (N + 1)
            logt = -np.log1p(1 / nlive)
        Wt = self.logw + logL + np.log1p(-np.exp(logt))
        self.logZ = np.logaddexp(self.logZ, Wt)
        # Update information estimate
        if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
            info = (
                np.exp(Wt - self.logZ) * logL
                + np.exp(oldZ - self.logZ) * (self.info[-1] + oldZ)
                - self.logZ
            )
            self.info.append(info)

        # Update history
        self.logw += logt
        self.logLs.append(logL)
        self.log_vols.append(self.logw)
        if self.track_gradients:
            self.gradients.append(
                (self.logLs[-1] - self.logLs[-2])
                / (self.log_vols[-1] - self.log_vols[-2])
            )

    def finalise(self):
        """
        Compute the final evidence with more accurate integrator
        Call at end of sampling run to refine estimate
        """
        # Trapezoidal rule
        # Extra point represents X=0 and assume max(L) = L[-1]
        self.logZ = log_integrate_log_trap(
            np.array(self.logLs + [self.logLs[-1]]),
            np.array(self.log_vols + [np.NINF]),
        )
        return self.logZ

    @nessai_style()
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
        plt.title(
            f"log Z={self.logZ:.2f} "
            f"H={self.info[-1] * np.log2(np.e):.2f} bits"
        )
        plt.grid(which="both")
        plt.xlabel("log prior-volume")
        plt.ylabel("log-likelihood")
        plt.xlim([self.log_vols[-1], self.log_vols[0]])

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
            plt.close()
            logger.debug(f"Saved nested sampling plot as {filename}")
        else:
            return fig

    @property
    def log_posterior_weights(self):
        """Compute the log-posterior weights."""
        log_L = np.array(self.logLs + [self.logLs[-1]])
        log_vols = np.array(self.log_vols + [np.NINF])
        log_Z = log_integrate_log_trap(log_L, log_vols)
        log_w = logsubexp(log_vols[:-1], log_vols[1:])
        log_post_w = log_L[1:-1] + log_w[:-1] - log_Z
        return log_post_w
