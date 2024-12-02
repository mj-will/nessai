# -*- coding: utf-8 -*-
"""Base nested sampler object"""

import datetime
import logging
import os
import pickle
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union
from warnings import warn

import numpy as np
import torch
from glasflow import __version__ as glasflow_version

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
    checkpoint_callback : Callback
        Callback function to be used instead of the default function. The
        function will be called in the code:`checkpoint` method as:
        :code`checkpoint_callback(self)`.
    logging_interval : int, optional
        The interval in seconds used for periodic logging. If not specified,
        then periodic logging is disabled.
    log_on_iteration : bool
        If true logging will occur based on the iteration. If false logging
        will be periodic if `logging_interval` is set. In case where neither
        logging is enabled, `log_on_iteration` will be set to true with an
        interval of :code:`nlive`.
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
        rng: Optional[np.random.Generator] = None,
        checkpointing: bool = True,
        checkpoint_interval: int = 600,
        checkpoint_on_iteration: bool = False,
        checkpoint_callback: Optional[Callable] = None,
        logging_interval: int = None,
        log_on_iteration: bool = True,
        resume_file: str = None,
        plot: bool = True,
        n_pool: Optional[int] = None,
        pool: Optional[Any] = None,
    ):
        logger.info("Initialising nested sampler")

        self.info_enabled = logger.isEnabledFor(logging.INFO)
        self.debug_enabled = logger.isEnabledFor(logging.DEBUG)
        self.configure_rng(seed, rng)
        model.set_rng(self.rng)
        model.verify_model()
        self.n_pool = n_pool
        self.model = model
        self.model.configure_pool(pool=pool, n_pool=n_pool)

        self.nlive = nlive
        self.plot = plot
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_on_iteration = checkpoint_on_iteration
        self.checkpoint_callback = checkpoint_callback
        if self.checkpoint_on_iteration:
            self._last_checkpoint = 0
        else:
            self._last_checkpoint = datetime.datetime.now()
        self.sampling_time = datetime.timedelta()
        self.sampling_start_time = datetime.datetime.now()
        self.iteration = 0
        self.finalised = False
        self.resumed = False

        self.configure_output(output, resume_file=resume_file)
        self.configure_periodic_logging(logging_interval, log_on_iteration)

        self.live_points = None
        self.history = None

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

    @property
    @abstractmethod
    def posterior_effective_sample_size(self):
        """The effective sample size of the posterior distribution"""
        raise NotImplementedError()

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

    def update_output(self, output: str) -> None:
        """Update the output directory and resume file.

        Parameters
        ----------
        output: str
            Path to the output directory
        """
        self.output = output
        resume_file = os.path.split(self.resume_file)[1]
        self.resume_file = os.path.join(output, resume_file)

    def configure_rng(
        self,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Configure the random number generation.

        If a seed is not specified, a random seed is generated using the
        specified random number generator, or the default numpy random number
        if not specified. In the latter case, the seed is then used to seed
        the new numpy random number generator.

        ..versionadded:: 0.14.0

        Parameters
        ----------
        seed : Optional[int]
            The random seed. If not specified, a random seed is generated.
        rng: Optional[np.random.Generator]
            Random number generator. If not specified, the default numpy
            random number generator is used with the specified seed.
        """
        if seed is None:
            logger.debug("Seed not specified, generating random seed")
            if rng is None:
                seed = random.randint(0, min(np.iinfo(int).max, 2**32 - 1))
            else:
                seed = rng.integers(0, min(np.iinfo(int).max, 2**32 - 1))
        logger.debug(f"Setting random seed to {seed}")
        self.seed = seed
        if rng is None:
            logger.debug(
                f"No rng specified, using the default rng with seed={seed}."
            )
            rng = np.random.default_rng(self.seed)
        self.rng = rng
        torch.manual_seed(self.seed)

    def configure_random_seed(self, seed: Optional[int] = None) -> None:
        """Initialise the random seed

        ..deprecated:: 0.14.0
            Deprecated in favour of
            :py:meth:`~nessai.samplers.base.BaseNestedSampler.configure_rng`.

        Parameters
        ----------
        seed : Optional[int]
            The random seed. If not specified, a random seed is generated.
        """
        msg = (
            "`configure_random_seed` is deprecated and will be removed in a "
            "future release. Use `configure_rng` instead."
        )
        warn(msg, FutureWarning)
        self.configure_rng(seed=seed)

    def configure_periodic_logging(self, logging_interval, log_on_iteration):
        """Configure the periodic logging.

        Parameters
        ----------
        logging_interval : int, optional
            The interval in seconds used for periodic logging. If not
            specified, then periodic logging is disabled.
        log_on_iteration : bool
            If true logging will occur based on the iteration. If false logging
            will be periodic if `logging_interval` is set. In case where
            neither logging is enabled, `log_on_iteration` will be set to true
            with an interval of :code:`nlive`.
        """
        self.logging_interval = logging_interval
        self.log_on_iteration = log_on_iteration
        if not self.logging_interval and not self.log_on_iteration:
            logger.warning(
                "All logging disabled. Enabling logging on iteration"
            )
            self.log_on_iteration = True
        if self.log_on_iteration:
            if self.logging_interval is None:
                self.logging_interval = self.nlive
            self._last_log = 0
        else:
            self._last_log = time.time()

    @abstractmethod
    def log_state(self):
        raise NotImplementedError()

    def periodically_log_state(self):
        """Log the state of the sampler.

        Calls :code:`log_state` if the elapsed interval in time (or iterations)
        is more than the specified interval.
        """
        if self.log_on_iteration:
            if not (self.iteration - self._last_log) >= self.logging_interval:
                return
            else:
                self._last_log = self.iteration
        else:
            now = time.time()
            if not (now - self._last_log) >= self.logging_interval:
                return
            else:
                self._last_log = now
        self.log_state()

    def initialise_history(self) -> None:
        """Initialise the dictionary to store history"""
        if self.history is None:
            logger.debug("Initialising history dictionary")
            self.history = dict(
                likelihood_evaluations=[],
                sampling_time=[],
                checkpoint_iterations=[],
            )
        else:
            logger.debug("History dictionary already initialised")

    def update_history(self) -> None:
        """Update the history dictionary"""
        self.history["likelihood_evaluations"].append(
            self.total_likelihood_evaluations
        )
        self.history["sampling_time"].append(
            self.current_sampling_time.total_seconds()
        )

    def checkpoint(
        self,
        periodic: bool = False,
        force: bool = False,
        save_existing: bool = True,
    ):
        """Checkpoint the classes internal state.

        Parameters
        ----------
        periodic : bool
            Indicates if the checkpoint is regular periodic checkpointing
            or forced by a signal. If forced by a signal, it will show up on
            the state plot.
        force : bool
            Force the sampler to checkpoint.
        save_existing : bool
            If True, the previous checkpoint will be copied to `.old`.
            If False, the new checkpoint will override the previous file.
        """
        now = datetime.datetime.now()
        if not periodic:
            if self.history:
                self.history["checkpoint_iterations"] += [self.iteration]
            else:
                logger.warning(
                    "Could not log checkpoint iteration in the history"
                )
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
        logger.info("Checkpointing nested sampling")
        if self.checkpoint_callback:
            self.checkpoint_callback(self)
        else:
            safe_file_dump(
                self, self.resume_file, pickle, save_existing=save_existing
            )
        self.sampling_start_time = datetime.datetime.now()

    @classmethod
    def resume_from_pickled_sampler(
        cls,
        sampler: Any,
        model: Model,
        output: Optional[str] = None,
        checkpoint_callback: Optional[Callable] = None,
    ):
        """Resume from pickle data.

        Parameters
        ----------
        data : Any
            Pickle data
        model : :obj:`nessai.model.Model`
            User-defined model
        output : Optional[str]
            New output directory. If not specified, the output directory will
            be the same as the previous run.
        checkpoint_callback : Optional[Callable]
            Checkpoint callback function. If not specified, the default method
            will be used.

        Returns
        -------
        Instance of BaseNestedSampler
        """
        logger.info(f"Resuming instance of {cls.__name__}")
        if model.rng is None:
            model.set_rng(sampler.rng)
        model.likelihood_evaluations += (
            sampler._previous_likelihood_evaluations
        )
        model.likelihood_evaluation_time += datetime.timedelta(
            seconds=sampler._previous_likelihood_evaluation_time
        )
        if output is not None and output != sampler.output:
            logger.info(
                f"Overwriting output from {sampler.output} to {output}"
            )
            os.makedirs(output, exist_ok=True)
            sampler.update_output(output)
        sampler.model = model
        sampler.resumed = True
        sampler.checkpoint_callback = checkpoint_callback
        return sampler

    @classmethod
    def resume(
        cls,
        filename: str,
        model: Model,
        output: Optional[str] = None,
        **kwargs,
    ):
        """Resumes the interrupted state from a checkpoint pickle file.

        Parameters
        ----------
        filename : str
            Pickle file to resume from
        model : :obj:`nessai.model.Model`
            User-defined model
        output : Optional[str]
            New output directory. If not specified, the output directory will
            be the same as the previous run.

        Returns
        -------
        obj
            Instance of BaseNestedSampler
        """
        logger.info(f"Resuming {cls.__name__} from {filename}")
        with open(filename, "rb") as f:
            sampler = pickle.load(f)
        return cls.resume_from_pickled_sampler(
            sampler, model, output=output, **kwargs
        )

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
        d["glasflow_version"] = glasflow_version
        d["seed"] = self.seed
        d["sampling_time"] = self.sampling_time.total_seconds()
        d["total_likelihood_evaluations"] = self.model.likelihood_evaluations
        d["likelihood_evaluation_time"] = (
            self.likelihood_evaluation_time.total_seconds()
        )
        d["history"] = self.history
        if hasattr(self.model, "truth"):
            d["truth"] = self.model.truth
        return d

    def __getstate__(self):
        d = self.__dict__
        exclude = {"model", "proposal", "checkpoint_callback"}
        state = {k: d[k] for k in d.keys() - exclude}
        state["_previous_likelihood_evaluations"] = d[
            "model"
        ].likelihood_evaluations
        state["_previous_likelihood_evaluation_time"] = d[
            "model"
        ].likelihood_evaluation_time.total_seconds()
        return state
