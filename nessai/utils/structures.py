# -*- coding: utf-8 -*-
"""
Utilities for manipulating python structures such as lists and dictionaries.
"""


def replace_in_list(target_list, targets, replacements):
    """
    Replace (in place) an entry in a list with a given element.

    Parameters
    ----------
    target_list : list
        List to update
    targets : list
        List of items to update
    replacements : list
        List of replacement items
    """
    if not isinstance(targets, list):
        targets = [targets]
    if not isinstance(replacements, list):
        replacements = [replacements]

    if not len(targets) == len(replacements):
        raise RuntimeError('Targets and replacements are different lengths!')

    if not all([t in target_list for t in targets]):
        raise ValueError(f'Targets {targets} not in list: {target_list}')

    for t, r in zip(targets, replacements):
        i = target_list.index(t)
        target_list[i] = r
