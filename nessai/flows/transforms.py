# -*- coding: utf-8 -*-
"""Transform for use in normalising flows."""
from nflows.transforms import InverseTransform, Sigmoid


class Logit(InverseTransform):
    """Logit transform that has learn temperature as an option."""
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__(
            Sigmoid(
                temperature=temperature,
                eps=eps,
                learn_temperature=learn_temperature,
            )
        )
