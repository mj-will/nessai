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

## Commit messages

We follow the same guidelines as SciPy, see the [SciPy documentation](https://docs.scipy.org/doc/scipy/dev/contributor/development_workflow.html#writing-the-commit-message). This includes the use of acronyms at the beginning of commit messages, see below for a complete list.
An example commit message:

```
EHN: add support for Neural Spline Flows

More details can also be added after a blank line, this could include a
reference to an open issue or another commit.
```

### Standard commit acronyms

These are based on the acronyms specified in the [SciPy guidelines](https://docs.scipy.org/doc/scipy/dev/contributor/development_workflow.html#writing-the-commit-message) with some
additions

```
API: an (incompatible) API change
BLD: change related to building nessai
BUG: bug fix
CI: changes to the continuous integration
DEP: deprecate something, or remove a deprecated object
DEV: development tool or utility
DOC: documentation
ENH: enhancement
MAINT: maintenance commit (refactoring, typos, etc.)
REV: revert an earlier commit
STY: style fix (whitespace, PEP8)
TST: addition or modification of tests
REL: related to releasing nessai
```

## Testing nessai

When contributing code to `nessai` please ensure that you also contribute corresponding unit tests and integration tests where applicable. We test `nessai` using `pytest` and strive to test all of the core functionality in `nessai`. Tests should be contained with the `tests` directory and follow the naming convention `test_<name>.py`. We also welcome improvements to the existing tests and testing infrastructure.

See the `pytest` [documentation](https://docs.pytest.org/) for further details on how to write tests using `pytest`.
