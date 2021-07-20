# Contributing to nessai

## Installation

To install ``nessai`` and contribute clone the repo and install the additional dependencies with:

```console
$ cd nessai
$ pip install -e .[dev]
```

## Format checking

We use [pre-commit](https://pre-commit.com/) to check the quality of code before committing, this includes checking code meets [PEP8](https://www.python.org/dev/peps/pep-0008/) style guidelines.

This requires some setup:

```console
$ pip install pre-commit # Should already be installed
$ cd nessai
$ pre-commit install
```

Now we you run `$ git commit` `pre-commit` will run a series of checks. Some checks will automatically change the code and others will print warnings that you must address and re-commit.

## Testing nessai

When contributing code to `nessai` please ensure that you also contribute corresponding unit tests and integration tests where applicable. We test `nessai` using `pytest` and strive to test all of the core functionality in `nessai`. Tests should be contained with the `tests` directory and follow the naming convention `test_<name>.py`. We also welcome improvements to the existing tests and testing infrastructure.

See the `pytest` [documentation](https://docs.pytest.org/) for further details on how to write tests using `pytest`.
