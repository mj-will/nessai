====
FAQs
====


When should I use :code:`allow_multi_valued_likelihood`?
--------------------------------------------------------

:code:`allow_multi_valued_likelihood` is a flag that can be set to `True` to allow
sampling with likelihood functions that are not deterministic, i.e. they do not
return the same value when called twice with the same point.

By default, :code:`allow_multi_valued_likelihood` is set to `False`. This means
that :code:`nessai` will raise an error if the likelihood function returns
different values for the same point:

.. code-block:: text

    Repeated calls to the log-likelihood with the same parameters returned different values.

Setting this flag to :code:`True` can be useful when using likelihood functions
that include, for example, a Monte Carlo integration step. However, it is
important to note that this can lead to incorrect results if the scale of the
variance on the likelihood for a single sample is comparable to the width
of the likelihood distribution for the posterior samples.

.. warning ::
    This flag should be used with caution and only when necessary. It can
    lead to very inefficient sampling and results that are entirely incorrect.


Why am I getting warnings about repeated log-likelihood values?
---------------------------------------------------------------

I you see the following warning:

.. code-block:: text

    Initial live points contain repeated log-likelihood values!
    This will likely lead to issues in the sampling process.

then it is likely that the log-likelihood function is returning the same value
for multiple points. This is not allowed in the standard nested sampling algorithm
and can lead to incorrect results.

One common cause of this error is the use of :code:`numpy.nan_to_num` to convert
:code:`-numpy.inf` to a finite value. :code:`nessai` is designed to handle
:code:`-numpy.inf` values and treat them as invalid regions of the parameter space.
It is therefore not recommended to use :code:`numpy.nan_to_num` in the
log-likelihood function.


Large dynamic range
-------------------

If you see an error such as:

.. code-block:: text

    Rescaling is not invertible for x! This may be due to the large dynamic range (log10 dynamic range=18.0, min=1.0377468396342845e-09, max=950413414.495942).

This may an indication that the dynamic range of the parameter is too large for the default floating point precession (float64).
In this case, it may be beneficial to apply a log transformation to the parameter before applying a reparameterisation such as :code:`scaleandshift` or :code:`rescaletobounds`.

This can be done using the `reparameterisations` keyword argument when calling the sampler. For example:

.. code-block:: python

    fs = FlowSampler(
        model=model,
        ...,
        reparameterisations={
            'log-standardise': ["x"],
        }
    )


The list should contain the name of the parameter(s) which large dynamic ranges.
This will apply a log transformation to the `x` parameter, before standardising the samples.
See :ref:`Configuring reparameterisations<Configuring reparameterisations>` for more details.
