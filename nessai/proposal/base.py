# -*- coding: utf-8 -*-
"""
Base object for all proposal classes.
"""
from abc import ABC, abstractmethod
import datetime
import logging

logger = logging.getLogger(__name__)


class Proposal(ABC):
    """
    Base proposal object

    Parameters
    ----------
    model: obj
        User-defined model
    """

    def __init__(self, model):
        self.model = model
        self.populated = True
        self._initialised = False
        self.training_count = 0
        self.population_acceptance = None
        self.population_time = datetime.timedelta()
        self.r = None
        self.samples = []
        self.indices = []
        self._checked_population = True

    @property
    def initialised(self):
        """Boolean that indicates if the proposal is initialised or not."""
        return self._initialised

    @initialised.setter
    def initialised(self, boolean):
        """Setter for initialised

        If value is set to true, the proposal method is tested with `test_draw`
        """
        if boolean:
            self._initialised = boolean
            # TODO: make this usable
            # self.test_draw()
        else:
            self._initialised = boolean

    def initialise(self):
        """
        Initialise the proposal
        """
        self.initialised = True

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

    def test_draw(self):
        """
        Test the draw method to ensure it returns a sample in the correct
        format and the the log prior is computed.
        """
        logger.debug(f"Testing {self.__class__.__name__} draw method")

        test_point = self.model.new_point()
        new_point = self.draw(test_point)

        if new_point["logP"] != self.model.log_prior(new_point):
            raise RuntimeError("Log prior of new point is incorrect!")

        logger.debug(f"{self.__class__.__name__} passed draw test")

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
