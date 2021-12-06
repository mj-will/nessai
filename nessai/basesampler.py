# -*- coding: utf-8 -*-
"""Base nested sampler object"""
from abc import ABC, abstractmethod
import datetime
import logging
import os
import pickle
from typing import Union

import numpy as np
import torch

from . import __version__ as version
from .model import Model
from .utils import safe_file_dump

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
    resume_file : str, optional
        Name of the file the sampler will be saved to and resumed from.
    plot : bool, optional
        Boolean to enable or disable plotting.
    """
    def __init__(
        self,
        model: Model,
        nlive: int,
        output: str = None,
        seed: int = None,
        checkpointing: bool = True,
        resume_file: str = None,
        plot: bool = True,
    ):
        logger.info('Initialising nested sampler')

        self.info_enabled = logger.isEnabledFor(logging.INFO)
        model.verify_model()
        self.model = model

        self.nlive = nlive
        self.plot = plot
        self.checkpointing = checkpointing
        self.sampling_time = datetime.timedelta()
        self.sampling_start_time = datetime.datetime.now()
        self.iteration = 0
        self.checkpoint_iterations = []

        self.configure_random_seed(seed)
        self.configure_output(output, resume_file=resume_file)

        self.live_points = None

    @property
    def current_sampling_time(self):
        if self.finalised:
            return self.sampling_time
        else:
            return self.sampling_time \
                    + (datetime.datetime.now() - self.sampling_start_time)

    def configure_output(
        self,
        output: Union[str, None],
        resume_file: Union[str, None] = None
    ):
        """
        Set up the output folder

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

    def configure_random_seed(self, seed: Union[int, None]):
        """
        Initialise the random seed
        """
        self.seed = seed
        if self.seed is not None:
            logger.debug(f'Setting random seed to {seed}')
            np.random.seed(seed=self.seed)
            torch.manual_seed(self.seed)

    def checkpoint(self, periodic: bool = False):
        """
        Checkpoint the classes internal state

        Parameters
        ----------
        periodic : bool
            Indicates if the checkpoint is regular periodic checkpointing
            or forced by a signal. If forced by a signal, it will show up on
            the state plot.
        """
        if not periodic:
            self.checkpoint_iterations += [self.iteration]
        self.sampling_time += \
            (datetime.datetime.now() - self.sampling_start_time)
        logger.critical('Checkpointing nested sampling')
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
        logger.critical('Resuming NestedSampler from ' + filename)
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        model.likelihood_evaluations += obj._previous_likelihood_evaluations
        obj.model = model
        return obj

    @abstractmethod
    def nested_sampling_loop(self):
        raise NotImplementedError()

    def get_result_dictionary(self):
        """Return a dictionary that contains results.

        Only includes vesrion, seed and sampling time. Child classes should
        call this method and add to the dictionary.
        """
        d = dict()
        d['version'] = version
        d['seed'] = self.seed
        d['sampling_time'] = self.sampling_time.total_seconds()
        if hasattr(self.model, 'truth'):
            d['truth'] = self.model.truth
        return d

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_previous_likelihood_evaluations'] = \
            state['model'].likelihood_evaluations
        del state['model']
        return state
