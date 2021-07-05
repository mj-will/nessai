# -*- coding: utf-8 -*-
import pytest
from unittest.mock import create_autospec

from nessai.proposal import FlowProposal


@pytest.fixture()
def proposal():
    return create_autospec(FlowProposal)
