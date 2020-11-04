# Contributing to nessai

## Installation

To install nessai and contribute clone the repo and install the additional dependenies with:

```console
$ cd nessai
$ pip install -e .[dev]
```

## Format checking

We use [pre-commit](https://pre-commit.com/) to check the quality of code before commiting, this includes checking code meets [PEP8](https://www.python.org/dev/peps/pep-0008/) style guidelines.

This requires some setup:

```console
$ pip install pre-commit # Should already be installed
$ cd nessai
$ pre-commit install
```

Now we you run `$ git commit` `pre-commit` will run a series of checks. Some checks will automatically change the code and others will print warnings that you must address and re-commit.
