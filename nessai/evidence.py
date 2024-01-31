# -*- coding: utf-8 -*-
"""
Functions related to computing the evidence.
"""
from abc import ABC, abstractmethod
import logging
from typing import Optional

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

    def get_logx_live_points(self, nlive: int):
        """Get the log-prior volume of the current live points.

        Computes the values assuming the run were to terminate at the current
        iteration.

        Parameters
        ----------
        nlive : int
            The number of live points.

        Returns
        -------
        numpy.ndarray
            Array of log-prior volumes of length :code:`nlive`.
        """
        nlive_per_iteration = np.arange(nlive, 0, -1, dtype=float)
        if self.expectation.lower() == "logt":
            logt = -1.0 / nlive_per_iteration
        elif self.expectation.lower() == "t":
            logt = -np.log1p(1.0 / nlive_per_iteration)
        log_x = self.logw + np.cumsum(logt)
        return log_x

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


class _INSIntegralState(_BaseNSIntegralState):
    """
    Object to handle computing the evidence for importance nested sampling.
    """

    def __init__(self) -> None:
        self._n_ns = 0
        self._n_lp = 0
        self._logZ = -np.inf
        self._weights_ns = None
        self._weights_lp = None
        self._weights = None

    @property
    def logZ(self) -> float:
        """The current log-evidence."""
        return self._logZ - np.log(self._n)

    log_evidence = logZ
    """Alias for logZ"""

    @property
    def log_evidence_error(self) -> float:
        """Alias for compute_uncertainty"""
        return self.compute_uncertainty(log_evidence=True)

    @property
    def log_evidence_live_points(self) -> float:
        """Log-evidence in the live points."""
        if self._weights_lp is None:
            raise RuntimeError("Live points are not set")
        return logsumexp(self._weights_lp) - np.log(self._weights_lp.size)

    @property
    def log_evidence_nested_samples(self):
        """Log-evidence in the nested samples"""
        return logsumexp(self._weights_ns) - np.log(self._weights_ns.size)

    @property
    def log_posterior_weights(self) -> np.ndarray:
        """Compute the log posterior weights.

        If the live points have been specified, then weights will be computed
        for these as well.
        """
        return self._weights - self.logZ

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
        self._weights_ns = nested_samples["logL"] + nested_samples["logW"]
        if live_points is not None:
            self._weights_lp = live_points["logL"] + live_points["logW"]
            self._weights = np.concatenate(
                [
                    self._weights_ns,
                    self._weights_lp,
                ]
            )
        else:
            self._weights = self._weights_ns
            self._weights_lp = None
        self._logZ = logsumexp(self._weights)
        self._n = self._weights.size

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

    def compute_uncertainty(self, log_evidence: bool = True) -> float:
        """Compute the uncertainty on the current estimate of the evidence.

        Parameters
        ----------
        log_evidence :
            If true, return the uncertainty for the log-evidence rather than
            the evidence.
        """
        n = self._n
        Z_hat = np.exp(self.logZ, dtype=np.longdouble)
        Z = np.exp(self._weights, dtype=np.longdouble)
        # Standard error sqrt(Var[Z] / n)
        u = np.sqrt(np.sum((Z - Z_hat) ** 2) / (n * (n - 1)))
        if log_evidence:
            # sigma[ln Z] = |sigma[Z] / Z|
            return float(np.abs(u / Z_hat))
        else:
            return u


def log_evidence_from_ins_samples(samples: np.ndarray) -> float:
    """Compute the log-evidence for a given set of samples from the importance
    nested sampler.

    Parameters
    ----------
    samples : numpy.ndarray
        Array of samples from the importance nested samples. Must have the logW
        field.

    Returns
    -------
    float
        The log-evidence
    """
    return logsumexp(samples["logL"] + samples["logW"]) - np.log(len(samples))
