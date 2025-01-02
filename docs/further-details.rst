===============
Further details
===============

Using live points
=================

Live points in ``nessai`` are stored in numpy structured arrays. These are array have fields which can be accessed like the values of dictionaries but they can also be indexed.
``nessai`` include various functions for constructing live point arrays and also converting these arrays to other common python formats, see :py:mod:`nessai.livepoint`

Here's an example of constructing a live point:

.. ipython:: python

    from nessai.livepoint import parameters_to_live_point
    x = parameters_to_live_point([1, 2], ['a', 'b'])
    print(x)    # the live point
    print(x.dtype.names)
    print(x['a'])    # the value of parameter 'a'


Importantly the structured arrays used for live points can also contain multiple live points:

.. ipython:: python

    from nessai.livepoint import numpy_array_to_live_points
    import numpy as np
    rng = np.random.default_rng(42)
    x = rng.random((10, 2))    # 10 live points with 2 parameters
    print(x)
    y = numpy_array_to_live_points(x, ['a', 'b'])    # call the parameters a and b
    print(y)
    y['a']    # all the values of a
    y[0]      # the first live point


Using analytic priors
=====================

``nessai`` includes the option for sampling from analytic priors. This should be enabled when the method :py:meth:`nessai.model.Model.new_point` draws directly from the priors. This eliminates the need for additional rejection sampling to ensure new points are correctly distributed.

To use this setting, the user must re-define ``new_point`` when defining the model as described in :doc:`running the sampler<running-the-sampler>`. This method must return samples as live points, see :ref:`using live points<Using live points>`. Once the method is redefined, set :code:`analytic_priors=True` when calling :py:class:`~nessai.flowsampler.FlowSampler`.


Checkpointing and resuming
==========================

Both the standard and importance nested samplers support checkpointing and
resuming. By default, the samplers periodically checkpoint to pickle file based
on the time elapsed since the last checkpoint. This behaviour can be configured
via various keyword arguments.


Configuration
-------------

The following options are available in all the sampler classes:

* :python:`checkpointing: bool`: Boolean to toggle checkpointing. If false, the sampler will not periodically checkpoint but will checkpoint at the end of sampling.
* :python:`checkpoint_on_iteration: bool`: Boolean to enable checkpointing based on the number of iterations rather than the elapsed time.
* :python:`checkpoint_interval: int`: The interval between checkpointing, the units depend on the value of :python:`checkpoint_interval`; if it false, is value the interval is specified in seconds; if it is true, the interval is specified in iterations.
* :python:`checkpoint_callback: Callable`: Callback function to be used instead of the default function. See `Checkpoint callbacks`_ for more details.

The following options are available when creating an instance of
:py:class:`~nessai.flowsampler.FlowSampler`:

* :python:`resume: bool`: Boolean to entirely enable or disable resuming irrespective of if there is a file or data to resume from.
* :python:`resume_file: str`: Name of the resume file.
* :python:`resume_data: Any`: Data to resume the sampler from instead of a resume file. The data will be passed to the :python:`resume_from_pickled_sampler` of the relevant class.


Resuming a sampling run
-----------------------

A sampling run can be resumed from either an existing resume file, which is
loaded automatically, or by specifying pickled data to resume from.
We recommended using the resume files, which are produced automatically, for
most applications.

The recommended method for resuming a run is by calling :py:class:`~nessai.flowsampler.FlowSampler` with
the same arguments that were originally used to start run; ensuring
:python:`resume=True` and :python:`resume_file` matches the name of the
:code:`.pkl` file in the output directory (the default is
:code:`nested_sampler_resume.pkl`).

.. note::

    Depending on how the sampling was interrupted, some progress may be lost and
    the sampling may resume from an earlier iteration.

Alternatively, you can specify the :python:`resume_data` argument which takes
priority over the resume file.
This will be passed to the :python:`resume_from_pickled_sampler` of the
corresponding sampler class.

.. note::

    If the output directory has been moved, make sure to change the
    :code`output` argument when calling :code:`FlowSampler`. The sampler
    will then automatically update the relevant paths.


Checkpoint callbacks
--------------------

Checkpoint callbacks allow the user to specify a custom function to use for
checkpointing the sampler.
This allows, for example, for the sampler to checkpoint an existing file rather.

The checkpoint callback function will be called in the :code:`checkpoint` method
with the class instance as the only argument, i.e.
:python:`checkpoint_callback(self)`.

All the sampler classes define custom :py:meth:`~nessai.samplers.base.BaseNestedSampler.__getstate__` methods that are
compatible with pickle and can be used to obtain a pickled representation of
the state of the sampler. Below is an example of a valid callback

.. code-block:: python

    import pickle
    filename = "checkpoint.pkl"

    def checkpoint_callback(state):
        with open(filename, "wb") as f:
            pickle.dump(state, f)

This could then passed as a keyword argument when running or resuming a sampler
via :py:class:`~nessai.flowsampler.FlowSampler`.

.. warning::
    The checkpoint callback is not included in the output of :python:`__getstate__`
    and must be specified when resuming the sampler via :py:class:`~nessai.flowsampler.FlowSampler`.
