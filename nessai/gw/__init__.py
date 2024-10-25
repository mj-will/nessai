# -*- coding: utf-8 -*-
"""
Functions and classes for using nessai for gravitational-wave inference.
"""

import warnings

from .proposal import GWFlowProposal

__all__ = ["GWFlowProposal"]

warnings.warn(
    (
        "The `nessai.gw` module will be deprecated in the next release in "
        "favour of the nessai-gw package. This packages provides the same "
        "functionality as`nessai.gw` via the plugin interface."
        "For more details, see: https://github.com/mj-will/nessai-gw"
    ),
    FutureWarning,
)
