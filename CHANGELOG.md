# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

These changes will be included in the next release.


## [0.2.3] - 2021-02-24

Add support for Python >= 3.6 and other minor changes and bug fixes

### Added

- Badges for DOI and PyPI versions.
- Add support for Python >= 3.6.
- Improve doc-strings and tweak settings for doc-strings in the documentation.
- Add tests for plotting functions.
- Added sections to README and docs on citing `nessai`.

### Changed

- Remove `:=` operator to enable support for Python >= 3.6.
- Plotting functions are now more constent and all return the figure if `filename=None`.

### Fixed

- Fixed bug when plotting non-structured arrays with `plot_1d_comparison` and specifying `parameters`.
- Fixed bug where `plot_indices` failed if using an empty array but worked with an empty list.

### Removed

- Remove `plot_posterior` because functionality is include in `plot_live_points`.
- Remove `plot_likelihood_evaluations` because information is already contained in the state plot.
- Remove `plot_acceptance` as it is only by agumented proposel which is subject to change.
- Remove `plot_flow`.

## [0.2.2] - 2021-02-19

This release was added to trigger Zenodo for producing a DOI.

### Added

- Docs badge

## [0.2.1] - 2021-02-18

Minor repository related fixes. Core code remains unchanged.

### Added

- PyPI workflow to automatically release package to PyPI

### Fixed

- Fixed issue with README not rendering of PyPi

## [0.2.0] - 2021-02-18

First public release.

### Added

- Complete documentation
- Use `setup.cfg` and `pyproject.toml` for installing package
- `reparemeterisations` submodule for more specific reparameterisations
- `half_gaussian.py` example

### Changed

- Change to use `main` instead of `master`
- Default `GWFlowProposal` changed to used `reparameterisations`
- Split `proposal.py` into various submodules
- Minor updates to examples
- `max_threads` default changed to 1.

### Fixed

- Fix a bug where `maximum_uninformed` did not have the expected behaviour.

### Deprecated

- Original `GWFlowProposal` method renamed to `LegacyGWFlowProposal`. Will be removed in the next release.

[Unreleased]: https://github.com/mj-will/nessai/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/mj-will/nessai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/mj-will/nessai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/mj-will/nessai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mj-will/nessai/compare/v0.1.1...v0.2.0
