"""Configuration for the reparameterisation tests"""
from dataclasses import dataclass
from typing import List

import pytest


@dataclass
class _LightReparam:
    """Reparameterisation-like object"""

    name: str
    parameters: List[str]
    requires: List[str]


@pytest.fixture
def LightReparam() -> _LightReparam:
    return _LightReparam
