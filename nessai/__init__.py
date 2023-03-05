# -*- coding: utf-8 -*-
"""nessai: Nested sampling with Artificial Intelligence

nessai is a nested sampling algorithm for Bayesian Inference that incorporates
normalising flows. It is designed for applications where the Bayesian
likelihood is computationally expensive.
"""
import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())
