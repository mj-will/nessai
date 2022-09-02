# -*- coding: utf-8 -*-
"""
Utilities for configuring torch.
"""
import logging

import torch

logger = logging.getLogger(__name__)

dtype_mapping = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def set_torch_default_dtype(dtype) -> torch.dtype:
    """Set the default dtype for torch tenors.

    If dtype is None, returns the default dtype.

    Parameters
    ----------
    dtype : {"float32", "float64", torch.dtype}
        The new default dtype for torch.

    Returns
    -------
    torch.dtype
        The torch dtype used to set the default.
    """
    if dtype is None:
        return torch.get_default_dtype()
    if isinstance(dtype, str):
        if dtype not in dtype_mapping:
            raise ValueError(
                f"Unknown torch dtype: {dtype}. "
                f"Choose from: {list(dtype_mapping.keys())}"
            )
        dtype = dtype_mapping.get(dtype)
    logger.info(f"Setting torch dtype to {dtype}")
    torch.set_default_dtype(dtype)
    return dtype
