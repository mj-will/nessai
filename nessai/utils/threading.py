# -*- coding: utf-8 -*-
"""
Utilities related to managing threads used by nessai.
"""
import logging

import torch

logger = logging.getLogger(__name__)


def configure_threads(max_threads=None, pytorch_threads=None, n_pool=None):
    """Configure the number of threads available.

    This is necessary when using PyTorch on the CPU as by default it will use
    all available threads.

    Notes
    -----
    Uses ``torch.set_num_threads``. If pytorch threads is None but other
    arguments are specified then the value is inferred from them.

    Parameters
    ----------
    max_threads: int, optional
        Maximum total number of threads to use between PyTorch and
        multiprocessing.
    pytorch_threads: int, optional
        Maximum number of threads for PyTorch on CPU.
    n_pool: int, optional
        Number of pools to use if using multiprocessing.
    """
    if max_threads is not None:
        if pytorch_threads is not None and pytorch_threads > max_threads:
            raise RuntimeError(
                f'More threads assigned to PyTorch ({pytorch_threads}) '
                f'than are available ({max_threads})'
            )
        if n_pool is not None and n_pool >= max_threads:
            raise RuntimeError(
                f'More threads assigned to pool ({n_pool}) than are '
                f'available ({max_threads})'
            )
        if (n_pool is not None and pytorch_threads is not None and
                (pytorch_threads + n_pool) > max_threads):
            raise RuntimeError(
                f'More threads assigned to PyTorch ({pytorch_threads}) '
                f'and pool ({n_pool}) than are available ({max_threads})'
            )

    if pytorch_threads is None:
        if max_threads is not None:
            if n_pool is not None:
                pytorch_threads = max_threads - n_pool
            else:
                pytorch_threads = max_threads

    if pytorch_threads is not None:
        logger.debug(
            f'Setting maximum number of PyTorch threads to {pytorch_threads}')
        torch.set_num_threads(pytorch_threads)
