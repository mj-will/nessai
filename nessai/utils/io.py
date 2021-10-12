# -*- coding: utf-8 -*-
"""
Utilities related to loading files, saving files etc.
"""
import json
import os
import shutil

import numpy as np

from ..livepoint import live_points_to_dict


def is_jsonable(x):
    """Check if an object is JSON serialisable.

    Based on: https://stackoverflow.com/a/53112659

    Parameters
    ----------
    x : obj
        Object to check

    Returns
    -------
    bool
        Boolean that indicates if the object is JSON serialisable.
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class NessaiJSONEncoder(json.JSONEncoder):
    """Class to encode numpy arrays and other non-serialisable objects.

    Based on: https://stackoverflow.com/a/57915246.

    Examples
    --------
    This class should be used in the ``cls`` argument::

        with open(filename, 'w') as wf:
             json.dump(d, wf, indent=4, cls=NessaiJSONEncoder)
    """

    def default(self, obj):
        """Method that returns a serialisable object"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif not is_jsonable(obj):
            return str(obj)
        else:
            return super().default(obj)


def safe_file_dump(data, filename, module, save_existing=False):
    """Safely dump data to a .pickle file.

    See Bilby for the original impletmentation:
    https://git.ligo.org/michael.williams/bilby/-/blob/master/bilby/core/utils.py

    Parameters
    ----------
    data :
        Data to dump.
    filename : str
        The file to dump to.
    module : {pickle, dill}
        The python module to use.
    save_existing : bool, optional
        If true move the existing file to <file>.old.
    """
    if save_existing:
        if os.path.exists(filename):
            old_filename = filename + ".old"
            shutil.move(filename, old_filename)

    temp_filename = filename + ".temp"
    with open(temp_filename, "wb") as file:
        module.dump(data, file)
    shutil.move(temp_filename, filename)


def save_live_points(live_points, filename):
    """Save live points to a file using JSON.

    Live points are converted to a dictionary and then saved.

    Parameters
    ----------
    live_points : ndarray
        Live points to save.
    filename : str
        File to save to.
    """
    d = live_points_to_dict(live_points)
    with open(filename, 'w') as wf:
        json.dump(d, wf, indent=4, cls=NessaiJSONEncoder)
