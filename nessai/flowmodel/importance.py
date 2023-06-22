# -*- coding: utf-8 -*-
"""
FlowModel for use in importance nested sampling.
"""
import copy
import glob
import logging
import os
from typing import Optional

import numpy as np
import torch

from .base import FlowModel
from .utils import update_model_config

from ..flows import configure_model


logger = logging.getLogger(__name__)


class ImportanceFlowModel(FlowModel):
    """Flow Model that contains multiple flows for importance sampler."""

    models: torch.nn.ModuleList = None
    _resume_n_models: int = None

    def __init__(self, config=None, output=None):
        super().__init__(config=config, output=output)
        self.weights_files = []
        self.models = torch.nn.ModuleList()

    @property
    def model(self):
        """The current flow (model).

        Returns None if the no models have been added.
        """
        if self.models:
            return self.models[-1]
        else:
            logger.warning("Model not defined yet!")
            return None

    @model.setter
    def model(self, model):
        if model is not None:
            self.models.append(model)

    @property
    def n_models(self) -> int:
        """Number of models (flows)"""
        if self.models:
            return len(self.models)
        else:
            return 0

    def initialise(self) -> None:
        """Initialise things"""
        self.initialised = True

    def reset_optimiser(self) -> None:
        """Reset the optimiser to point at current model.

        Uses the original optimiser and kwargs.
        """
        self._optimiser = self.get_optimiser(
            optimiser=self.optimiser, **self.optimiser_kwargs
        )

    def add_new_flow(self, reset=False):
        """Add a new flow"""
        logger.debug("Add a new flow")
        if reset or not self.models:
            new_flow, self.device = configure_model(self.model_config)
        else:
            new_flow = copy.deepcopy(self.model)
        logger.debug(f"Training device: {self.device}")
        self.inference_device = torch.device(
            self.model_config.get("inference_device_tag", self.device)
            or self.device
        )
        logger.debug(f"Inference device: {self.inference_device}")
        self.models.eval()
        self.models.append(new_flow)
        self.reset_optimiser()

    def log_prob_ith(self, x, i):
        """Compute the log-prob for the ith flow"""
        x = (
            torch.from_numpy(x)
            .type(torch.get_default_dtype())
            .to(self.models[i].device)
        )
        if self.models[i].training:
            self.models[i].eval()
        with torch.no_grad():
            log_prob = self.models[i].log_prob(x)
        log_prob = log_prob.cpu().numpy().astype(np.float64)
        return log_prob

    def log_prob_all(self, x, exclude_last=False):
        """Compute the log probability using all of the stored models."""
        x = (
            torch.from_numpy(x)
            .type(torch.get_default_dtype())
            .to(self.model.device)
        )
        if self.models.training:
            self.models.eval()
        n = self.n_models
        if exclude_last:
            n -= 1
        if n <= 0:
            raise RuntimeError(
                "Cannot exclude last when flow there is only one flow"
            )
        log_prob = torch.empty(x.shape[0], n)
        with torch.no_grad():
            for i, m in enumerate(self.models[:n]):
                log_prob[:, i] = m.log_prob(x)
        log_prob = log_prob.cpu().numpy().astype(np.float64)
        return log_prob

    def sample_ith(self, i, N=1):
        """Draw samples from the ith flow"""
        if self.models is None:
            raise RuntimeError("Models are not initialised yet!")
        if self.models[i].training:
            self.models[i].eval()

        with torch.no_grad():
            x = self.models[i].sample(int(N))

        x = x.cpu().numpy().astype(np.float64)
        return x

    def save_weights(self, weights_file) -> None:
        """Save the weights file."""
        super().save_weights(weights_file)
        self.weights_files.append(self.weights_file)

    def load_all_weights(self) -> None:
        """Load all of the weights files for each flow.

        Resets any existing models.
        """
        self.models = torch.nn.ModuleList()
        logger.debug(f"Loading weights from {self.weights_files}")
        for wf in self.weights_files:
            new_flow, self.device = configure_model(self.model_config)
            new_flow.load_state_dict(torch.load(wf))
            self.models.append(new_flow)
        self.models.eval()

    def update_weights_path(
        self, weights_path: str, n: Optional[int] = None
    ) -> None:
        """Update the weights path.

        Searches in the specified directory for weights files.

        Parameters
        ----------
        weights_path : str
            Path to the directory that contains the weights files.
        n : Optional[int]
            The number of files to load. If not specified, :code:`n_models` is
            used instead. Must be specified when resuming since the models list
            is not saved.
        """
        all_weights_files = glob.glob(
            os.path.join(weights_path, "", "level_*", "model.pt")
        )

        if n is None:
            if self.n_models:
                n = self.n_models
            else:
                raise RuntimeError(
                    "n is None and no models are defined, cannot update "
                    "weights path."
                )

        logger.debug(f"Loading weights from: {all_weights_files}")
        if len(all_weights_files) < n:
            raise RuntimeError(
                f"Cannot use weights from: {weights_path}. Not enough files."
            )
        elif len(all_weights_files) > n:
            logger.warning(
                "More weights files than expected. Some files will be skipped."
            )
        self.weights_files = [
            os.path.join(weights_path, f"level_{i}", "model.pt")
            for i in range(n)
        ]

    def resume(
        self, model_config: dict, weights_path: Optional[str] = None
    ) -> None:
        """Resume the model"""
        self.model_config = update_model_config(model_config)
        if weights_path is None:
            logger.debug(
                "Not weights path specified, looking in output directory"
            )
            weights_path = self.output
        self.update_weights_path(weights_path, n=self._resume_n_models)
        self.load_all_weights()
        self.initialise()

    def __getstate__(self):
        d = self.__dict__
        # Avoid making a copy because models can be large and this doubles the
        # memory usage.
        exclude = {"models", "_optimiser", "model_config"}
        state = {k: d[k] for k in d.keys() - exclude}
        state["initialised"] = False
        state["models"] = None
        state["_resume_n_models"] = len(d["models"])
        return state
