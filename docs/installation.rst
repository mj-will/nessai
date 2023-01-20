============
Installation
============

.. note::

    It is recommended to install PyTorch before installing ``nessai``. See the `PyTorch documentation <https://pytorch.org/>`_ for how to install the correct version.

.. tabs::

     .. code-tab:: console conda

        conda install -c conda-forge -c pytorch nessai

     .. code-tab:: console pip

        pip install nessai


This will install the dependencies needed but the version of PyTorch will not necessarily be the correct version for your system.


Nessai for development
======================

To install ``nessai`` for development purposes see the `contribution guidelines <https://github.com/mj-will/nessai/blob/master/CONTRIBUTING.md>`_.


Using nessai with bilby
=======================

``nessai`` is included in ``bilby`` version 1.1.0 onwards. For instructions on how to install ``bilby`` see `here <https://lscsoft.docs.ligo.org/bilby/index.html>`_.

For examples of how to run ``nessai`` via ``bilby`` see the `examples directory <https://github.com/mj-will/nessai/tree/main/examples>`_.
