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


def save_to_json(d, filename, **kwargs):
    """Save a dictionary to a JSON file.

    Kwargs are passed to :code:`json.dump`. Uses :code:`NessaiJSONEncoder` by
    default.

    Parameters
    ----------
    d : dict
        Dictionary to save.
    filename : str
        Filename (with the extension) to save the dictionary to. Should include
        the complete path.
    kwargs : Any
        Keyword arguments passed to :code:`json.dump`.
    """
    default_kwargs = dict(
        indent=4,
        cls=NessaiJSONEncoder,
    )
    default_kwargs.update(kwargs)
    with open(filename, "w") as fp:
        json.dump(d, fp, **default_kwargs)


def safe_file_dump(data, filename, module, save_existing=False):
    """Safely dump data to a .pickle file.

    See Bilby for the original implementation.

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
    with open(filename, "w") as wf:
        json.dump(d, wf, indent=4, cls=NessaiJSONEncoder)


def encode_for_hdf5(value):
    """Encode a value for HDF5 file format.

    Parameters
    ----------
    value : Any
        Value to encode.

    Returns
    -------
    Any
        Encoded value.
    """
    if value is None:
        output = "__none__"
    else:
        output = value
    return output


def add_dict_to_hdf5_file(hdf5_file, path, d):
    """Save a dictionary to a HDF5 file.

    Based on :code:`recursively_save_dict_contents_to_group` in bilby.

    Parameters
    ----------
    hdf5_file : h5py.File
        HDF5 file.
    path : str
        Path added to the keys of the dictionary.
    d : dict
        The dictionary to save.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            add_dict_to_hdf5_file(hdf5_file, path + key + "/", value)
        else:
            hdf5_file[path + key] = encode_for_hdf5(value)


def save_dict_to_hdf5(d, filename):
    """Save a dictionary to a HDF5 file.

    Parameters
    ----------
    d : dict
        Dictionary to save.
    filename : str
        Filename (with the extension) to save the dictionary to. Should include
        the complete path.
    """
    import h5py

    with h5py.File(filename, "w") as f:
        add_dict_to_hdf5_file(f, "/", d)
