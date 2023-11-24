# -*- coding: utf-8 -*-
"""
Functions related to computing the posterior samples.
"""
import logging

import numpy as np
from scipy.special import logsumexp

from .evidence import logsubexp, log_integrate_log_trap
from .utils.stats import effective_sample_size

logger = logging.getLogger(__name__)


def compute_weights(samples, nlive, expectation="logt"):
    """
    Returns the log-evidence and log-weights for the log-likelihood samples
    assumed to the result of nested sampling with nlive live points

    Parameters
    ----------
    samples : array_like
        Log-likelihood samples.
    nlive : Union[int, array_like]
        Number of live points used in nested sampling.
    expectation : str, {logt, t}
        Method used to compute the expectation value for the shrinkage t.
        Choose between log <t> or <log t>. Defaults to <log t>.

    Returns
    -------
    float
        The computed log-evidence.
    array_like
        Array of computed weights (already normalised by the log-evidence).
    """
    samples = np.asarray(samples)

    if isinstance(nlive, (int, float)):
        nlive_per_iteration = nlive * np.ones_like(samples)
        nlive_per_iteration[-nlive:] = np.arange(nlive, 0, -1, dtype=float)
    else:
        if len(nlive) != len(samples):
            raise ValueError("nlive and samples are different lengths")
        nlive_per_iteration = nlive.copy()

    if expectation.lower() == "logt":
        logt = -1.0 / nlive_per_iteration
    elif expectation.lower() == "t":
        logt = -np.log1p(1.0 / nlive_per_iteration)
    else:
        raise ValueError(f"Expectation must be t or logt, got: {expectation}")

    # One point at X=1 and X=0
    n_vols = len(samples) + 2
    log_vols = np.zeros(n_vols)
    log_vols[1:-1] = np.cumsum(logt)
    log_vols[-1] = np.NINF

    # First point represents the entire prior
    # Last point represents X=0 and assume max(L) = L[-1]
    log_likelihoods = np.concatenate(
        [np.array([np.NINF]), samples, np.array([samples[-1]])]
    )

    log_evidence = log_integrate_log_trap(log_likelihoods, log_vols)

    log_w = logsubexp(log_vols[:-1], log_vols[1:])

    log_post_w = log_likelihoods[1:-1] + log_w[:-1]
    log_post_w -= log_evidence

    return log_evidence, log_post_w


def draw_posterior_samples(
    nested_samples,
    nlive=None,
    n=None,
    log_w=None,
    method="rejection_sampling",
    return_indices=False,
    expectation="logt",
):
    """Draw posterior samples given the nested samples.

    Requires either the posterior weights or the number of live points.

    Parameters
    ----------
    nested_samples : structured array
        Array of nested samples.
    nlive : int, optional
        Number of live points used during nested sampling. Either this
        arguments or log_w must be specified.
    n : int, optional
        Number of samples to draw. Only used for importance sampling. If not
        specified, the effective sample size is used instead.
    log_w : array_like, optional
        Array of posterior weights. If specified the weights are not computed
        and these weights are used instead.
    method : str
        Method for drawing the posterior samples. Choose from

            - :code:`'rejection_sampling'`
            - :code:`'multinomial_resampling'`
            - :code:`'importance_sampling'` (same as multinomial)
    return_indices : bool
        If true return the indices of the accepted samples.
    expectation : str, {logt, t}
        Method used to compute the expectation value for the shrinkage t.
        Choose between log <t> or <log t>. Defaults to <log t>. Only used when
        :code:`log_w` is not specified.

    Returns
    -------
    posterior_samples : numpy.ndarray
        Samples from the posterior distribution.
    indices : numpy.ndarray
        Indices of the accepted samples in the original nested samples.
        Only returned if :code:`return_indices` is True.

    Raises
    ------
    ValueError
        If the chosen method is not a valid method.
    """
    nested_samples = np.asarray(nested_samples)
    if log_w is None:
        _, log_w = compute_weights(
            nested_samples["logL"], nlive, expectation=expectation
        )
    else:
        log_w = np.asarray(log_w)
    ess = effective_sample_size(log_w)
    logger.info(f"Effective sample size: {ess:.1f}")
    if method == "rejection_sampling":
        logger.info("Producing posterior samples using rejection sampling")
        if n is not None:
            logger.warning(
                "Number of samples cannot be specified for rejection sampling"
            )
        log_w = log_w - np.max(log_w)
        n_expected = np.exp(logsumexp(log_w))
        logger.info(f"Expect {n_expected} samples from rejection sampling")
        log_u = np.log(np.random.rand(nested_samples.size))
        indices = np.where(log_w > log_u)[0]
        samples = nested_samples[indices]
    elif method in ["importance_sampling", "multinomial_resampling"]:
        logger.info("Producing posterior samples using multinomial resampling")
        if n is None:
            n = int(ess)
        log_w = log_w - logsumexp(log_w)
        indices = np.random.choice(
            nested_samples.size, size=n, p=np.exp(log_w), replace=True
        )
        samples = nested_samples[indices]
    else:
        raise ValueError(
            f"Unknown method of drawing posterior samples: {method}"
        )

    if return_indices:
        return samples, indices
    else:
        return samples
