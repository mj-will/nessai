# -*- coding: utf-8 -*-
"""
Base object for all proposal classes.
"""

import datetime
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Proposal(ABC):
    """
    Base proposal object

    Parameters
    ----------
    model: obj
        User-defined model
    rng: np.random.Generator, optional
        Random number generator. If not provided, a new generator is created.
    """

    def __init__(self, model, rng: Optional[np.random.Generator] = None):
        self.model = model
        if rng is None:
            logger.debug("No rng specified, using the default rng.")
            rng = np.random.default_rng()
        self.rng = rng
        self.populated = True
        self._initialised = False
        self.training_count = 0
        self.population_acceptance = None
        self.population_time = datetime.timedelta()
        self.r = np.nan
        self.samples = []
        self.indices = []
        self._checked_population = True

    @property
    def initialised(self):
        """Boolean that indicates if the proposal is initialised or not."""
        return self._initialised

    @initialised.setter
    def initialised(self, boolean):
        """Setter for initialised"""
        if boolean:
            self._initialised = boolean
        else:
            self._initialised = boolean

    def initialise(self):
        """
        Initialise the proposal
        """
        self.initialised = True

    def update_output(self, output: str) -> None:
        """
        Update the output directory.

        Only updates the output if the proposal has an output attribute.

        Parameters
        ----------
        output: str
            Path to the output directory
        """
        if hasattr(self, "output"):
            logger.debug(f"Updating output directory to {output}")
            self.output = output
            os.makedirs(self.output, exist_ok=True)
        else:
            logger.debug("No output directory to update")

    def evaluate_likelihoods(self):
        """Evaluate the likelihoods for the pool of live points."""
        self.samples["logL"] = self.model.batch_evaluate_log_likelihood(
            self.samples
        )

    @abstractmethod
    def draw(self, old_param):
        """
        New a new point given the old point
        """
        raise NotImplementedError

    def train(self, x, **kwargs):
        """
        Train the proposal method

        Parameters
        ----------
        x: array_like
            Array of live points to use for training
        kwargs:
            Any of keyword arguments
        """
        logger.error("This proposal method cannot be trained")

    def resume(self, model):
        """
        Resume the proposal with the model
        """
        self.model = model

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["model"]
        return state
