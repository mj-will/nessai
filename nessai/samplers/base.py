# -*- coding: utf-8 -*-
"""Base nested sampler object"""
from abc import ABC, abstractmethod
import datetime
import logging
import os
import pickle
from typing import Any, Optional, Union

import numpy as np
import torch

from .. import __version__ as version
from ..model import Model
from ..utils import safe_file_dump

logger = logging.getLogger(__name__)


class BaseNestedSampler(ABC):
    """Base nested sampler class.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User-defined model
    output : str, optional
        Path for the output. If not specified the current working directory
        is used.
    seed : int, optional
        Seed used to seed numpy and torch.
    checkpointing : bool, optional
        Boolean to toggle checkpointing, must be enabled to resume the sampler.
        If false the sampler is still saved at the end of sampling.
    checkpoint_interval : int
        The interval used for checkpointing. By default this is a time interval
        in seconds. If :code:`checkpoint_on_iteration=True` this corresponds to
        the number of iterations between checkpointing.
    checkpoint_on_iteration : bool
        If true the checkpointing interval is checked against the number of
        iterations
    resume_file : str, optional
        Name of the file the sampler will be saved to and resumed from.
    plot : bool, optional
        Boolean to enable or disable plotting.
    n_pool : int, optional
        Number of threads to when for creating the multiprocessing pool.
    pool : object
        User defined multiprocessing pool that will be used when evaluating
        the likelihood.
    """

    def __init__(
        self,
        model: Model,
        nlive: int,
        output: str = None,
        seed: int = None,
        checkpointing: bool = True,
        checkpoint_interval: int = 600,
        checkpoint_on_iteration: bool = False,
        resume_file: str = None,
        plot: bool = True,
        n_pool: Optional[int] = None,
        pool: Optional[Any] = None,
    ):
        logger.info("Initialising nested sampler")

        self.info_enabled = logger.isEnabledFor(logging.INFO)
        model.verify_model()
        self.n_pool = n_pool
        self.model = model
        self.model.configure_pool(pool=pool, n_pool=n_pool)

        self.nlive = nlive
        self.plot = plot
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_on_iteration = checkpoint_on_iteration
        if self.checkpoint_on_iteration:
            self._last_checkpoint = 0
        else:
            self._last_checkpoint = datetime.datetime.now()
        self.sampling_time = datetime.timedelta()
        self.sampling_start_time = datetime.datetime.now()
        self.iteration = 0
        self.checkpoint_iterations = []
        self.finalised = False
        self.resumed = False

        self.configure_random_seed(seed)
        self.configure_output(output, resume_file=resume_file)

        self.live_points = None

    @property
    def current_sampling_time(self):
        if self.finalised:
            return self.sampling_time
        else:
            return self.sampling_time + (
                datetime.datetime.now() - self.sampling_start_time
            )

    @property
    def likelihood_evaluation_time(self):
        """Current log-likelihood time"""
        return self.model.likelihood_evaluation_time

    @property
    def total_likelihood_evaluations(self):
        """Total number of likelihood evaluations"""
        return self.model.likelihood_evaluations

    likelihood_calls = total_likelihood_evaluations
    """Alias for :code:`total_likelihood_evaluations`"""

    def configure_output(
        self, output: Union[str, None], resume_file: Union[str, None] = None
    ):
        """Configure the output folder

        Parameters
        ----------
        output : str
            Directory where the results will be stored
        resume_file : str, optional
            Specific file to use for checkpointing. If not specified the
            default is used (nested_sampler_resume.pkl)
        """
        if output is None:
            output = os.getcwd()
        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)

        if resume_file is None:
            resume_file = os.path.join(output, "nested_sampler_resume.pkl")
        else:
            resume_file = os.path.join(output, resume_file)

        self.output = output
        self.resume_file = resume_file

    def configure_random_seed(self, seed: Optional[int]):
        """Initialise the random seed.

        Parameters
        ----------
        seed : Optional[int]
            The random seed. If not specified, no seed is set.
        """
        self.seed = seed
        if self.seed is not None:
            logger.debug(f"Setting random seed to {seed}")
            np.random.seed(seed=self.seed)
            torch.manual_seed(self.seed)

    def checkpoint(self, periodic: bool = False, force: bool = False):
        """Checkpoint the classes internal state.

        Parameters
        ----------
        periodic : bool
            Indicates if the checkpoint is regular periodic checkpointing
            or forced by a signal. If forced by a signal, it will show up on
            the state plot.
        force : bool
            Force the sampler to checkpoint.
        """
        now = datetime.datetime.now()
        if not periodic:
            self.checkpoint_iterations += [self.iteration]
        elif force:
            pass
        else:
            if self.checkpoint_on_iteration:
                if (
                    self.iteration - self._last_checkpoint
                ) >= self.checkpoint_interval:
                    self._last_checkpoint = self.iteration
                else:
                    return
            else:
                if (
                    now - self._last_checkpoint
                ).total_seconds() >= self.checkpoint_interval:
                    self._last_checkpoint = now
                else:
                    return
        self.sampling_time += now - self.sampling_start_time
        logger.critical("Checkpointing nested sampling")
        safe_file_dump(self, self.resume_file, pickle, save_existing=True)
        self.sampling_start_time = datetime.datetime.now()

    @classmethod
    def resume(cls, filename: str, model: Model):
        """Resumes the interrupted state from a checkpoint pickle file.

        Parameters
        ----------
        filename : str
            Pickle file to resume from
        model : :obj:`nessai.model.Model`
            User-defined model
        Returns
        -------
        obj
            Instance of BaseNestedSampler
        """
        logger.critical("Resuming NestedSampler from " + filename)
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        model.likelihood_evaluations += obj._previous_likelihood_evaluations
        model.likelihood_evaluation_time += datetime.timedelta(
            seconds=obj._previous_likelihood_evaluation_time
        )
        obj.model = model
        obj.resumed = True
        return obj

    @abstractmethod
    def nested_sampling_loop(self):
        raise NotImplementedError()

    def close_pool(self, code=None):
        """Close the multiprocessing pool."""
        self.model.close_pool(code=code)

    def get_result_dictionary(self):
        """Return a dictionary that contains results.

        Only includes version, seed and sampling time. Child classes should
        call this method and add to the dictionary.
        """
        d = dict()
        d["version"] = version
        d["seed"] = self.seed
        d["sampling_time"] = self.sampling_time.total_seconds()
        d["total_likelihood_evaluations"] = self.model.likelihood_evaluations
        d[
            "likelihood_evaluation_time"
        ] = self.likelihood_evaluation_time.total_seconds()
        if hasattr(self.model, "truth"):
            d["truth"] = self.model.truth
        return d

    def __getstate__(self):
        d = self.__dict__
        exclude = {"model", "proposal"}
        state = {k: d[k] for k in d.keys() - exclude}
        state["_previous_likelihood_evaluations"] = d[
            "model"
        ].likelihood_evaluations
        state["_previous_likelihood_evaluation_time"] = d[
            "model"
        ].likelihood_evaluation_time.total_seconds()
        return state
