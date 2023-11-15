===============
Further details
===============

Using live points
=================

Live points in ``nessai`` are stored in numpy structured arrays. These are array have fields which can be accessed like the values of dictionaries but they can also be indexed. ``nessai`` include various functions for constructing live point arrays and also converting these arrays to other common python formats, see mod:`nessai.livepoint`

Here's an example of constructing a live point:

.. code-block:: python

    >>> from nessai.livepoint import parameters_to_live_point
    >>> x = parameters_to_live_point([1, 2], ['a', 'b'])
    >>> print(x)    # the live point
    (1., 2., 0., 0.)
    >>> print(x.dtype.names)
    ('a', 'b', 'logP', 'logL')
    >>> print(x['a'])    # the value of parameter 'a'
    1.0


Importantly the structured arrays used for live points can also contain multiple live points:

.. code-block:: python

    >>> from nessai.livepoint import numpy_array_to_live_points
    >>> x = np.random.rand(10, 2)    # 10 live points with 2 parameters
    >>> print(x)
    [[0.72451217 0.1788154 ]
     [0.31549832 0.55898106]
     [0.74000576 0.73103116]
     [0.37362176 0.25791271]
     [0.61056168 0.05940721]
     [0.33988486 0.54106604]
     [0.82653691 0.14523437]
     [0.62390321 0.32606928]
     [0.21743918 0.23915047]
     [0.45478996 0.09699358]]
    >>> y = numpy_array_to_live_points(x, ['a', 'b'])    # call the parameters a and b
    >>> print(y)
    array([(0.72451217, 0.1788154 , 0., 0.), (0.31549832, 0.55898106, 0., 0.),
           (0.74000576, 0.73103116, 0., 0.), (0.37362176, 0.25791271, 0., 0.),
           (0.61056168, 0.05940721, 0., 0.), (0.33988486, 0.54106604, 0., 0.),
           (0.82653691, 0.14523437, 0., 0.), (0.62390321, 0.32606928, 0., 0.),
           (0.21743918, 0.23915047, 0., 0.), (0.45478996, 0.09699358, 0., 0.)],
          dtype=[('a', '<f8'), ('b', '<f8'), ('logP', '<f8'), ('logL', '<f8')])
    >>> y['a']    # all the values of a
    array([0.72451217, 0.31549832, 0.74000576, 0.37362176, 0.61056168,
           0.33988486, 0.82653691, 0.62390321, 0.21743918, 0.45478996])
    >>> y[0]      # the first live point
    (0.72451217, 0.1788154, 0., 0.)


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


Detailed explanation of outputs
===============================

.. note::
    This section has not been completed yet
