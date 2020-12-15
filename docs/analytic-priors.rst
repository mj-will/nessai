=====================
Using Analytic Priors
=====================

Optional methods
----------------

There are further methods that the user can re-define:

**Drawing new points**

The ``Model`` includes a method for drawing new samples ``new_point`` and computing the corresponding log-probability of drawing the sample ``new_point_log_prob``. This is used when populating the initial pool of points and during the inital *uninformed sampling* when the normalising flow is not used. By default these points are drawn uniformally within user-defined ``bounds`` but the user can choose to re-define these methods.

If the samples produced by ``new_point`` are drawn directly from the prior the sampler can be ran with the flag ``analytic_priors=True`` and this will improve the efficiency of the initial *uniformed sampling*.

NOTE: the the new samples returned by ``new_point`` must be a structured array with the correct fields, the functions in :py:mod:`nessai.livepoint` are useful here.
