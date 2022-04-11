# -*- coding: utf-8 -*-

import pytest
from unittest.mock import MagicMock, create_autospec
from nessai.model import Model
from nessai.proposal.importance import ImportanceFlowProposal


@pytest.fixture
def proposal():
    obj = create_autospec(ImportanceFlowProposal)
    obj.model = MagicMock(spec=Model)
    return obj
