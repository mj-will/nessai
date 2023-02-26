[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4550693.svg)](https://doi.org/10.5281/zenodo.4550693)
[![PyPI](https://img.shields.io/pypi/v/nessai)](https://pypi.org/project/nessai/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/nessai.svg)](https://anaconda.org/conda-forge/nessai)
[![Documentation Status](https://readthedocs.org/projects/nessai/badge/?version=latest)](https://nessai.readthedocs.io/en/latest/?badge=latest)
![license](https://anaconda.org/conda-forge/nessai/badges/license.svg)
![tests](https://github.com/mj-will/nessai/actions/workflows/tests.yml/badge.svg)
![int-tests](https://github.com/mj-will/nessai/actions/workflows/integration-tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/mj-will/nessai/branch/main/graph/badge.svg?token=O7SN167SK6)](https://codecov.io/gh/mj-will/nessai)

# nessai: Nested Sampling with Artificial Intelligence

``nessai`` (/ˈnɛsi/): Nested Sampling with Artificial Intelligence

``nessai`` is a nested sampling algorithm for Bayesian Inference that incorporates normalisings flows. It is designed for applications where the Bayesian likelihood is computationally expensive.

## Installation

``nessai`` can be installed using ``pip``:

```console
pip install nessai
```

or via ``conda``

```console
conda install -c conda-forge -c pytorch nessai
```

### PyTorch

By default the version of PyTorch will not necessarily match the drivers on your system, to install a different version with the correct CUDA support see the PyTorch homepage for instructions: https://pytorch.org/.

### Using ``bilby``

As of `bilby` version 1.1.0, ``nessai`` is now supported by default but it is still an optional requirement. See the [``bilby`` documentation](https://lscsoft.docs.ligo.org/bilby/index.html) for installation instructions for `bilby`

See the examples included with ``nessai`` for how to run ``nessai`` via ``bilby``.

## Documentation

Documentation is available at: [nessai.readthedocs.io](https://nessai.readthedocs.io/)


## Contributing

Please see the guidelines [here](https://github.com/mj-will/nessai/blob/master/CONTRIBUTING.md).


## Acknowledgements

The core nested sampling code, model design and code for computing the posterior in ``nessai`` was based on [`cpnest`](https://github.com/johnveitch/cpnest) with permission from the authors.

The normalising flows implemented in ``nessai`` are all either directly imported from [`nflows`](https://github.com/bayesiains/nflows/tree/master/nflows) or heavily based on it.

Other code snippets that draw on existing code reference the source in their corresponding doc-strings.

The authors also thank Christian Chapman-Bird, Laurence Datrier, Fergus Hayes, Jethro Linley and Simon Tait for their feedback and help finding bugs in ``nessai``.

## Citing

If you find ``nessai`` useful in your work please cite the DOI for this code and our paper:

```bibtex
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

@article{PhysRevD.103.103006,
  title = {Nested sampling with normalizing flows for gravitational-wave inference},
  author = {Williams, Michael J. and Veitch, John and Messenger, Chris},
  journal = {Phys. Rev. D},
  volume = {103},
  issue = {10},
  pages = {103006},
  numpages = {19},
  year = {2021},
  month = {May},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevD.103.103006},
  url = {https://link.aps.org/doi/10.1103/PhysRevD.103.103006}
}
```
