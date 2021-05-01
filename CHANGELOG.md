# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added code to catch errors when calling `plot_live_points` when `gwpy` is installed.
- Added tests for `_NSIntegralState`.
- Add code coverage upload
- Added an example of using unbounded priors, `bilby_unbounded_priors.py`
- Added `Rescale` reparameterisation that just rescales by a constant and does not require prior bounds. Also add
tests for this reparameterisation.
- Added more GW examples.
- Added tests for `AugmentedFlowProposal`.
- Added an example using `AugmentedFlowProposal`.
- Added eggbox example.
- Added an error if calling `FlowProposal.rejection_sampling` with `FlowProposal.truncate=True` but `worst_q=None`.

### Changed

- Plotting logX vs logL now returns the figure is `filename=None`
- `NestedSampler.plot_state` now has the keyword argument `filename` and the figure is only saved if it is specified.
- Changed name from `_NSintegralState` to `_NSIntegralState`.
- `nessai.model.Model` now inherits from `abc.ABC` and `log_prior` and `log_likelihood` are now `abstractmethods`. This prevents the class from being used without redefining those methods.
- Updated `AumgentedFlowProposal` to work with current version of `FlowProposal`
- Fix random seed unit tests.
- Improved `FlowProposal.reset` so that all attributes that are changed by calling `draw` are reset.

### Fixed

- Fixed a bug when plotting the state plot from a saved instance of the sampler where the sampling time was changed based on the current time.
- Fixed a bug when using `plot_trace`, `plot_1d_comparison` or `plot_live_points` with a single parameter
- Total sampling time is now correctly displayed when producing a state plot from a saved sampler.
- Fixed a bug when using unbounded priors related to `Model.verify_model`
- Fix inversion-split with `RescaleToBounds`
- Fixed `AugmentedGWFlowProposal`.
- Fix typo in `FlowProposal.set_poolsize_scale` when `acceptance=0`
- Fixed unintended behaviour when `rescale_parameters` is a list and `boundary_inversion=True`, where the code would try apply inversion to all parameters in `Model.names`.
- Fixed bug where `z` returned by `FlowProposal.rejection_sampling` was incorrect when using truncation (which is not recommended).


## [0.2.4] - 2021-03-08

This release includes a number of bug fixes, changes to make the `GWFlowProposal` consistent with `LegacyGWFlowProposal` and a number of new unit tests to improve test coverage.

### Added

- Add poolsize to `AnalyticProposal`
- Add a test for sampling with multiprocessing.
- Add a test for sampling with `AnalyticProposal` and `RejectionProposal`.
- Add a test for using the proposal methods with `n_pool`
- Add tests for reparameterisations.
- Add a test for comparing `GWFlowProposal` and `LegacyGWFlowProposal`.

### Changed

- Changed prime priors in `LegacyGWFlowProposal` to not update. This improves efficiency.
- Changes to the reparameterisations to the the proposal consistent with `LegacayGWFlowProposal`:
    - Use [-1, 1] when inversion is enabled but not applied
- Improved errors when reparameterisations are configured incorrectly.

### Fixed

- Fixed a bug with saving results when multiprocessing is enabled.
- Fixed a bug with `AnalyticProposal` introduced in the last release.
- Fixed a bug with resuming when using certain reparameterisations.


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

[Unreleased]: https://github.com/mj-will/nessai/compare/v0.2.4...HEAD
[0.2.4]: https://github.com/mj-will/nessai/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/mj-will/nessai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/mj-will/nessai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/mj-will/nessai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mj-will/nessai/compare/v0.1.1...v0.2.0
