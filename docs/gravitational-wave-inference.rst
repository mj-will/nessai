============================
Gravitational-wave inference
============================

Basic configuration
===================

``nessai`` includes a default configuration for gravitational-wave inference. This includes default reparameterisations for common parameters. This functionality is enabled by adding :code:`flow_class='gwflowproposal'` to the arguments for the sampler.


Marginalisation
===============

Distance marginalisation
-------------------------

In the original paper we tested ``nessai`` with and without distance marginalisation and found that it can work with either. With the updated defaults, sampling with distance marginalisation is typically faster.

Phase marginalisation
---------------------

The current defaults for ``nessai`` are not well suited to sampling without phase marginalisation and these result in inefficient sampling. We believe that revised settings for this case will help and we are actively investigating this.

Time marginalisation
--------------------

Time marginalisation as implemented in ``bilby`` has been tested and results show that it can reduce sampling time without negatively impacting convergence.

The ``time_jitter`` parameter that is added by ``bilby`` is treated as a periodic parameter by default, however it is not clear is this treatment is necessary.


Calibration parameters
======================

Basic tests including calibration parameters have been conducted and no major issues have been identified. Sampling is, as expected, slower but the overall convergence does not seem to be affected.

These parameters typically use Gaussian priors without hard prior bounds. By default, this is not an issue because the :code:`NullReparameterisation` is used. However, if using a custom set of reparameterisations this should be accounted for.


Examples
========

Examples of running ``nessai`` for gravitational-wave inference can be found in the GW examples directory `here <https://github.com/mj-will/nessai/tree/main/examples/gw>`_.
