# -*- coding: utf-8 -*-
"""
Utilities related to optimisation.
"""
import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


logger = logging.getLogger(__name__)


def optimise_meta_proposal_weights(
    samples: np.ndarray,
    log_q: np.ndarray,
    method: str = "SLSQP",
    options: Optional[dict] = None,
    initial_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Optimise the weights of the meta proposal.

    Uses :code:`scipy.optimize.minimize`.

    Parameters
    ----------
    samples
        Samples drawn from the initial meta proposal.
    log_q
        Array of log probabilities for each proposal for each sample.
    method
        Optimisation method to use. See scipy docs for details.
    options
        Dictionary of options for :code:`scipy.optimize.minimize`.
    """
    if options is None and method == "SLSQP":
        options = dict(ftol=1e-10)

    n_prop = log_q.shape[-1]
    counts = np.unique(samples["it"], return_counts=True)[1]
    if initial_weights is None:
        initial_weights = counts / counts.sum()
    else:
        initial_weights = initial_weights / initial_weights.sum()

    log_Z = logsumexp(samples["logL"] - samples["logQ"]) - np.log(len(samples))

    log_pr = samples["logL"] - log_Z
    log_pr -= logsumexp(log_pr)
    pr = np.exp(log_pr)

    def loss(weights):
        """Computes the KL"""
        log_Q = logsumexp(log_q, b=weights, axis=1)
        return -np.mean(pr * log_Q)

    # Weights must sum to one
    constraint = {"type": "eq", "fun": lambda x: 1 - x.sum()}

    logger.info("Starting optimisation")
    result = minimize(
        loss,
        initial_weights,
        constraints=constraint,
        bounds=n_prop * [(0, 1)],
        method=method,
        options=options,
    )
    logger.info("Finished optimisation")
    logger.debug(f"Final weights: {result.x}")

    return np.array(result.x)
