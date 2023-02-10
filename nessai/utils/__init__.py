# -*- coding: utf-8 -*-
"""
General utilities for nessai.
"""
from .distributions import get_multivariate_normal, get_uniform_distribution
from .hist import auto_bins
from .indices import bonferroni_correction, compute_indices_ks_test
from .io import (
    NessaiJSONEncoder,
    is_jsonable,
    safe_file_dump,
    save_dict_to_hdf5,
    save_live_points,
    save_to_json,
)
from .logging import setup_logger
from .rescaling import (
    configure_edge_detection,
    detect_edge,
    determine_rescaled_bounds,
    inverse_rescale_minus_one_to_one,
    inverse_rescale_zero_to_one,
    logit,
    rescale_minus_one_to_one,
    rescale_zero_to_one,
    rescaling_functions,
    sigmoid,
)
from .sampling import (
    compute_radius,
    draw_gaussian,
    draw_nsphere,
    draw_surface_nsphere,
    draw_truncated_gaussian,
    draw_uniform,
)
from .distance import compute_minimum_distances
from .stats import rolling_mean
from .structures import replace_in_list
from .threading import configure_threads


__all__ = [
    "BoxUniform",
    "MultivariateNormal",
    "NessaiJSONEncoder",
    "auto_bins",
    "bonferroni_correction",
    "compute_indices_ks_test",
    "compute_minimum_distances",
    "compute_radius",
    "configure_edge_detection",
    "configure_threads",
    "detect_edge",
    "determine_rescaled_bounds",
    "distance",
    "distributions",
    "draw_gaussian",
    "draw_nsphere",
    "draw_surface_nsphere",
    "draw_truncated_gaussian",
    "draw_uniform",
    "get_multivariate_normal",
    "get_uniform_distribution",
    "hist",
    "indices",
    "inverse_rescale_minus_one_to_one",
    "inverse_rescale_zero_to_one",
    "is_jsonable",
    "logit",
    "rolling_mean",
    "replace_in_list",
    "rescale_minus_one_to_one",
    "rescale_zero_to_one",
    "rescaling",
    "rescaling_functions",
    "safe_file_dump",
    "sampling",
    "save_dict_to_hdf5",
    "save_live_points",
    "save_to_json",
    "setup_logger",
    "sigmoid",
    "spatial",
    "structures",
    "threading",
]
