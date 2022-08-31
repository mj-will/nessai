# -*- coding: utf-8 -*-
"""
Utilities for configuring torch.
"""
import logging
from typing import Literal, Union

import torch

logger = logging.getLogger(__name__)

dtype_mapping = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def set_torch_default_type(
    dtype: Union[Literal["float32", "float64"], torch.dtype, None],
) -> torch.dtype:
    """Set the default dtype for torch tenors.

    If dtype is None, returns the default dtype.

    Parameters
    ----------
    dtype
        The new default dtype for torch.

    Returns
    -------
    The torch dtype used to set the default.
    """
    if dtype is None:
        return torch.get_default_dtype()
    if isinstance(dtype, str):
        dtype = dtype_mapping.get(dtype)
        if dtype is None:
            raise ValueError(
                f"Unknown torch dtype: {dtype}. "
                f"Choose from: {list(dtype_mapping.keys())}"
            )
    logger.info(f"Setting torch dtype to {dtype}")
    torch.set_default_dtype(dtype)
    return dtype
