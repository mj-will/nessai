.. nessai documentation master file, created by
   sphinx-quickstart on Wed Sep 16 13:23:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nessai's documentation!
========================================

======
Nessai
======

``nessai`` (/ˈnɛsi/): Nested Sampling with Artificial Intelligence

``nessai`` is a nested sampling algorithm for Bayesian Inference that incorporates normalising flows. It is designed for applications where the Bayesian likelihood is computationally expensive.

The code is available at: https://github.com/mj-will/nessai.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   running-the-sampler
   sampler-configuration
   reparameterisations
   normalising-flows-configuration
   parallelisation
   gravitational-wave-inference
   further-details
   API reference </autoapi/nessai/index>

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   gaussian-example
   bilby-example

=============
Citing nessai
=============

If you find ``nessai`` useful in your work please cite the DOI for this code and our paper:

.. code-block:: bibtex

    @software{nessai,
      author       = {Michael J. Williams},
      title        = {nessai: Nested Sampling with Artificial Intelligence},
      month        = feb,
      year         = 2021,
      publisher    = {Zenodo},
      version      = {latest},
      doi          = {10.5281/zenodo.4550693},
      url          = {https://doi.org/10.5281/zenodo.4550693}
    }

    @article{williams2021nested,
      title={Nested Sampling with Normalising Flows for Gravitational-Wave Inference},
      author={Michael J. Williams and John Veitch and Chris Messenger},
      year={2021},
      eprint={2102.11056},
      archivePrefix={arXiv},
      primaryClass={gr-qc}
    }
