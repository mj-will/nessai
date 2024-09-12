from unittest.mock import MagicMock

from nessai.evidence import _INSIntegralState
from nessai.samplers.importancesampler import (
    ImportanceNestedSampler as INS,
)
from nessai.samplers.importancesampler import (
    OrderedSamples,
)


def test_log_posterior_weights_property(ins):
    ins._ordered_samples = MagicMock(spec=OrderedSamples)
    ins._ordered_samples.state = MagicMock(spec=_INSIntegralState)
    assert (
        INS.log_posterior_weights.__get__(ins)
        is ins._ordered_samples.state.log_posterior_weights
    )
