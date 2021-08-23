# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `in_bounds`, `parameter_in_bounds` and `sample_parameter` methods to `nessai.model.Model`.
- Implemented the option to specify the cosmology in `nessai.gw.utils.ComovingDistanceConverter` using `astropy`. Previously changing the value had no effect of the transformation.
- Add `'logit'` to the default reparameterisations ([!98](https://github.com/mj-will/nessai/pull/98))
- Add example using the Rosenbrock likelihood in two dimensions ([!99](https://github.com/mj-will/nessai/pull/99))
- Add a `colours` argument to `nessai.plot.plot_1d_comparison`
- Explicitly support Python 3.9 (Added Python 3.9 to unit tests)


### Changed

- `nessai.gw.utils.DistanceConverter` now inherits from `abc.ABC` and `to_uniform_parameter` and `from_uniform_parameter` are both abstract methods.
- `nessai.proposal.rejection.RejectionProposal` now inherits from `nessai.proposal.analytic.AnalyticProposal`. Functionality is the same but the code will be easier to maintain since this removes several methods that were identical.
- `nessai.proposal.base.Proposal` now inherits from `abc.ABC` and `draw` is an abstract method.
- `noise_scale='adaptive'` option in `FlowModel` now correctly uses a standard deviation of 0.2 times the mean nearest neighbour separation as described in [Moss 2019](https://arxiv.org/abs/1903.10860). Note that this feature is disabled by default, so this does not change the default behaviour.
- Refactor `nessai.utils` into a submodule.
- Change behaviour of `determine_rescaled_bounds` so that `rescale_bounds` is ignored when `inversion=True`. This matches the behaviour in `RescaledToBounds` where when boundary inversion is enabled, values are rescaled to $[0, 1]$ and then if no inversion if applied, changed to $[-1, 1]$.
- Tweaked `detect_edges` so that `both` is returned in cases where the lower and upper regions contain zero probability.
- `NestedSampler` no longer checks capitalisation of `flow_class` when determining which proposal class to use. E.g. `'FlowProposal'` and `'flowproposal'` are now both valid values.
- `NestedSampler.configure_flow_proposal` now raises `ValueError` instead of `RuntimeError` if `flow_class` is an invalid string.
- Raise a `ValueError` if `nessai.plot.plot_1d_comparison` is called with a labels list and the length does not match the number of sets of live points being compared.
- `nessai.flow.base.BaseFlow` now also inherits from `abc.ABC` and methods that should be defined by the user are abstract methods.
- Changed default to `fuzz=1e-12` in `nessai.utils.rescaling.logit` and `nessai.utils.rescaling.sigmoid` and improved stability.

### Fixed

- Fixed a typo in `nessai.gw.utils.NullDistanceConverter.from_uniform_parameter` that broke the method.
- Fixed a bug in `nessai.reparameterisations.RescaleToBounds` when using `offset=True` and `pre_rescaling` where the prime prior bounds were incorrectly set. ([!97](https://github.com/mj-will/nessai/pull/97))
- Fixed a bug that prevented disabling periodic checkpointing.
- Fixed a bug when calling `nessai.plot.plot_1d_comparison` with live points that contain a field with only infinite values.
- Fixed the log Jacobian determinant for `nessai.utils.rescaling.logit` and `nessai.utils.rescaling.sigmoid` which previously did not include the Jacobian for the fuzz when it was used.


## [0.3.0] Testing, testing and more testing - 2021-07-05

This release contains a large number of changes related to bugs and issues that were discovered when writing more tests for `nessai`.

It also adds a number of feature and examples.

**Note:** Runs produced with previous releases are incompatible with this release and cannot be resumed with out manual intervention.

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
- Add option to train using dataloaders or directly with tensors. This is faster when using CUDA.
- Add options to train with different optimisers: Adam, AdamW, SGD
- Add tests for `NestedSampler`
- Explicitly check prior bounds when using reparameterisations. This catches cases where infinite bounds are used and break some reparameterisations. (!82)
- Add error when calling `FlowProposal.populate` without initialising the proposal.
- Add `NestedSampler.plot_insertion_indices` to allow for easier plotting of insertion indices.
- Add `filename` keyword argument to `NestedSampler.plot_trace`.
- Added `batch_norm_within_layers` to `NeuralSplineFlow`

### Changed

- Plotting logX vs logL now returns the figure is `filename=None`
- `NestedSampler.plot_state` now has the keyword argument `filename` and the figure is only saved if it is specified.
- Changed name from `_NSintegralState` to `_NSIntegralState`.
- `nessai.model.Model` now inherits from `abc.ABC` and `log_prior` and `log_likelihood` are now `abstractmethods`. This prevents the class from being used without redefining those methods.
- Updated `AumgentedFlowProposal` to work with current version of `FlowProposal`
- Fix random seed unit tests.
- Improved `FlowProposal.reset` so that all attributes that are changed by calling `draw` are reset.
- Move `_NSIntegralState` and some functions from `posterior.py` to `evidence.py`
- `NestedSampler.check_flow_model_reset` will now NOT reset the flow it has never been trained (i.e `proposal.training_count==0`)
- Moved all legacy gw functions to `nessai/gw/legacy.py` and removed them from the coverage report.
- Minor improvements to `NestedSampler`
- Better handling on NaNs in `NestedSampler.populate_live_points`
- Minor improvements to plotting in `FlowProposal` and moved plotting to separate methods in `FlowProposal`.
- Switch to using `os.path.join` when joins paths.
- Improved `FlowProposal.reset`
- Renamed `FlexibleRealNVP` to `RealNVP`, shouldn't affect most uses since the default way to specify a flow is via strings in `configure_model`.
- Renamed `nessai.flows.utils.setup_model` to `configure_model`.
- Renamed `nessai.flows.utils.CustomMLP` to `MLP`
- Changed default value for `tail_bound` in `NeuralSplineFlow` to 5.


### Fixed

- Fixed a bug when plotting the state plot from a saved instance of the sampler where the sampling time was changed based on the current time.
- Fixed a bug when using `plot_trace`, `plot_1d_comparison` or `plot_live_points` with a single parameter
- Total sampling time is now correctly displayed when producing a state plot from a saved sampler.
- Fixed a bug when using unbounded priors related to `Model.verify_model`
- Fix inversion-split with `RescaleToBounds`
- Fixed `AugmentedGWFlowProposal`.
- Fixed a bug with `plot_live_points` when the hue parameter (`c`) was constant.
- Fixed a bug with the reparameterisation `Rescale` when `scale` was set to a negative number.
- Fixed a bug where `scale` could not be changed in `ToCartesian`.
- Fixed a error when specifying `NullReparameterisation` (!82)
- Fix typo in `FlowProposal.set_poolsize_scale` when `acceptance=0`
- Fixed unintended behaviour when `rescale_parameters` is a list and `boundary_inversion=True`, where the code would try apply inversion to all parameters in `Model.names`.
- Fixed bug where `z` returned by `FlowProposal.rejection_sampling` was incorrect when using truncation (which is not recommended).
- Fix `prior_sampling`
- Fixed minor typos in `nessai.proposal.flowproposal.py`


### Removed

- Remove "clip" option in `FlowProposal`, this was unused and untested.


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


[Unreleased]: https://github.com/mj-will/nessai/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/mj-will/nessai/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/mj-will/nessai/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/mj-will/nessai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/mj-will/nessai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/mj-will/nessai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mj-will/nessai/compare/v0.1.1...v0.2.0
