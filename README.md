[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4550694.svg)](https://doi.org/10.5281/zenodo.4550694) ![PyPI](https://img.shields.io/pypi/v/nessai) [![Documentation Status](https://readthedocs.org/projects/nessai/badge/?version=latest)](https://nessai.readthedocs.io/en/latest/?badge=latest)

# Nessai: Nested Sampling with Artificial Intelligence

``nessai`` (/ˈnɛsi/): Nested Sampling with Aritificial Intelligence

``nessai`` is a nested sampling algorithm for Bayesian Inference that incorporates normalisings flows. It is designed for applications where the Bayesian likelihood is computationally expensive.

## Installation

``nessai`` can be installed using ``pip``:

```console
$ pip install nessai
```

Installing via ``conda`` is not currently supported.

### PyTorch

By default the version of PyTroch will not necessarily match the drivers on your system, to install a different version with the correct CUDA support see the PyTorch homepage for instructions: https://pytorch.org/.

### Adding Bilby

This package requieres a fork of Bilby that includes the sampler, it can be installed by running (this requires pip):

```console
$ pip install git+https://git.ligo.org/michael.williams/bilby.git@add-nessai-sampler#egg=bilby
```

## Documentation

Documenation is available at: [nessai.readthedocs.io](https://nessai.readthedocs.io/)


## Contributing

Please see the guidelines [here](https://github.com/mj-will/nessai/blob/master/CONTRIBUTING.md).


## Acknowledgements

The core nested sampling code, model design and code for computing the posterior in ``nessai`` was based on [`cpnest`](https://github.com/johnveitch/cpnest) with permission from the authors.

The normalising flows implemented in ``nessai`` are all either directly imported from [`nflows`](https://github.com/bayesiains/nflows/tree/master/nflows) or heavily based on it.

Other code snippets that draw on existing code reference the source in their corresponding doc-strings.

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

@article{williams2021nested,
  title={Nested Sampling with Normalising Flows for Gravitational-Wave Inference},
  author={Michael J. Williams and John Veitch and Chris Messenger},
  year={2021},
  eprint={2102.11056},
  archivePrefix={arXiv},
  primaryClass={gr-qc}
}
```
