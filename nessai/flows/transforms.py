# -*- coding: utf-8 -*-
"""Transform for use in normalising flows."""
from nflows.transforms import LULinear as BaseLULinear
import torch


class LULinear(BaseLULinear):
    """Wrapper for LULinear from nflows that works with CUDA.

    The original implementation has a bug that prevents use with CUDA. See
    https://github.com/bayesiains/nflows/pull/38 for details.

    This should be removed if the bug is fixed in nflows.
    """

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(
            self.features, self.features, device=self.lower_entries.device)
        lower_inverse, _ = torch.triangular_solve(
            identity, lower, upper=False, unitriangular=True
        )
        weight_inverse, _ = torch.triangular_solve(
            lower_inverse, upper, upper=True, unitriangular=False
        )
        return weight_inverse
