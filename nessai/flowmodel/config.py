# -*- coding: utf-8 -*-
"""
Default configuration of FlowModel.
"""

from dataclasses import dataclass
from typing import Callable

from ..config import _BaseConfig


@dataclass
class FlowConfig(_BaseConfig):
    """Default configuration for flows."""

    # Use a dataclass since dictionaries are mutable
    n_inputs: int = None
    n_neurons: int = None
    n_blocks: int = 4
    n_layers: int = 2
    ftype: str = "RealNVP"
    flow: Callable = None
    distribution: str = None
    distribution_kwargs: dict = None


@dataclass
class TrainingConfig(_BaseConfig):
    """Default configuration for training."""

    # Use a dataclass since dictionaries are mutable
    device_tag: str = "cpu"
    inference_device_tag: str = None
    lr: float = 0.001
    annealing: bool = False
    clip_grad_norm: float = 5.0
    batch_size: int = 1000
    val_size: float = 0.1
    max_epochs: int = 500
    patience: int = 20
    noise_type: str = None
    noise_scale: float = None
    use_dataloader: bool = False
    optimiser: str = "adamw"
    optimiser_kwargs: dict = None


flow = FlowConfig()
training = TrainingConfig()
