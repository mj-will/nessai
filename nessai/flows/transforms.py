# -*- coding: utf-8 -*-
"""Transform for use in normalising flows."""
from warnings import warn
from glasflow.nflows.transforms import LULinear as BaseLULinear


class LULinear(BaseLULinear):
    """Wrapper for LULinear from nflows that works with CUDA.

    The original implementation has a bug that prevents use with CUDA. See
    https://github.com/bayesiains/nflows/pull/38 for details.

    This should be removed if the bug is fixed in nflows.
    """

    msg = (
        "`nessai.flows.transforms.LULinear` is deprecated and will be removed "
        "in a future release. Use `glasflow.nflows.transforms.LULinear` "
        "instead. "
    )
    warn(msg, FutureWarning)
