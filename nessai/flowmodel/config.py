# -*- coding: utf-8 -*-
"""
Default configuration of FlowModel.
"""
DEFAULT_FLOW_CONFIG = dict(
    n_inputs=None,
    n_neurons=None,
    n_blocks=4,
    n_layers=2,
    ftype="RealNVP",
    flow=None,
    distribution=None,
    distribution_kwargs=None,
)

DEFAULT_TRAINING_CONFIG = dict(
    device_tag="cpu",
    inference_device_tag=None,
    lr=0.001,
    annealing=False,
    clip_grad_norm=5,
    batch_size=1000,
    val_size=0.1,
    max_epochs=500,
    patience=20,
    noise_type=None,
    noise_scale=None,
    use_dataloader=False,
    optimiser="adamw",
    optimiser_kwargs=None,
)
