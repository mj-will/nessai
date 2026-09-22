# Contributing to nessai

## Installation

Use [Pixi](https://pixi.prefix.dev/latest/installation/) for development.
Environments and tasks are defined in `pyproject.toml`. Clone the repository,
then run:

```console
$ cd nessai
$ pixi install
$ pixi run pre-commit install
```

The default environment uses Python 3.11, CPU PyTorch, an editable nessai install,
and development, test, and gravitational-wave dependencies. Linux, macOS, and
Windows are supported. Dependencies are resolved locally; do not commit
`pixi.lock` or `.pixi/`. After changing dependencies, run `pixi install` again.

Use `pixi run <command>` or `pixi shell` to work inside the environment.
The `py39`, `py310`, `py311`, `py312`, and `py313` environments reproduce the CI
Python matrix. The `bilby` environment tests the published nessai-bilby plugin,
and `docs` builds the documentation.

## Format checking

We use [pre-commit](https://pre-commit.com/) to check the quality of code before committing, this includes checking code meets [PEP8](https://www.python.org/dev/peps/pep-0008/) style guidelines.

This requires some setup:

```console
$ pixi run pre-commit install
```

Now when you run `$ git commit` `pre-commit` will run a series of checks. Some checks will automatically change the code and others will print warnings that you must address and re-commit.

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

Run the checks and tests with:

```console
$ pixi run lint
$ pixi run format-check
$ pixi run test
$ pixi run test-integration
$ pixi run test-slow-integration
$ pixi run -e py39 test
$ pixi run -e bilby test-bilby
$ pixi run -e docs docs
```

Pass pytest arguments after the task, for example
`pixi run test tests/test_model.py`. Unit tests exclude integration tests;
run the two integration tasks separately when validating sampler changes.
