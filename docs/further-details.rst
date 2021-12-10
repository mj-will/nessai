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


Detailed explanation of outputs
===============================

.. note::
    This section has not been completed yet
