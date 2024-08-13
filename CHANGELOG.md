# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.2]

### Fixed

- Handle error when `linear_transform='None'` which occurs when using `bilby_pipe` after the `flow_config` changes (https://github.com/mj-will/nessai/pull/414)

## [0.13.1]

### Changed

- Make tests that require `faiss` are optional in the test suite
(https://github.com/mj-will/nessai/pull/408)


## [0.13.0]

### Added

- Add p-value and additional panel to indices plot
(https://github.com/mj-will/nessai/pull/391)
- Add support for numpy 2.0 (https://github.com/mj-will/nessai/pull/387)
- Add support for arbitrary priors in unit hypercube with the importance sampler
(https://github.com/mj-will/nessai/pull/377)

### Changed

- Simplify rescaling/reparameterisation configuration
https://github.com/mj-will/nessai/pull/395)
- The default reparameterisation has been changed from rescale-to-bounds to
z-score standardisation (https://github.com/mj-will/nessai/pull/395)
- Change default seaborn style to avoid plotting issues on some systems
(https://github.com/mj-will/nessai/pull/397)
- Rework flow configuration to use `flow_config` and `training_config` keyword
arguments (https://github.com/mj-will/nessai/pull/394)
- Skip nested sampling loop and populating live points if run is already finalised
(https://github.com/mj-will/nessai/pull/393, https://github.com/mj-will/nessai/pull/400)

### Deprecated

- Specifying `model_config` in the `flow_config` dictionary is now deprecated
(https://github.com/mj-will/nessai/pull/394)
- `FlowProposal.names`, `FlowProposal.rescaled_names` and
`FlowProposal.rescaled_dims` are now deprecated
(https://github.com/mj-will/nessai/pull/395)

### Removed

- `rescale_parameters`, `boundary_inversion`, `inversion_type`, `rescale_bounds`
`update_bounds`, `detect_edges`, `detect_edges_kwargs`,
have all been removed in favour of using the reparameterisations directly
(https://github.com/mj-will/nessai/pull/395)
- Drop support for Python 3.8 (https://github.com/mj-will/nessai/pull/396)


### Experimental

- Add experimental `ClusteringFlowProposal`
(https://github.com/mj-will/nessai/pull/380)
- Add experimental support for using flows directly from `glasflow`
(https://github.com/mj-will/nessai/pull/386)


## [0.12.0]

This release reworks large parts of the importance nested sampler to enable
drawing i.i.d samples during sampling.

The high-level API remains unchanged but the APIs for the
`ImportanceNestedSampler` and `ImportanceFlowProposal` classes have changed.
Existing runs of the importance nested sampler cannot be resumed with this
version.

### Added

- Add option to accumulate weights during rejection sampling ([#358](https://github.com/mj-will/nessai/pull/358))
- Add option to draw i.i.d samples during sampling when using the importance nested sampler ([#362](https://github.com/mj-will/nessai/pull/362))
- Add the `OrderedSamples` class for handling samples in the importance nested sampler ([#362](https://github.com/mj-will/nessai/pull/362))
- Add the `in_unit_hypercube` and `sample_unit_hypercube` methods to the model class `Model` ([#362](https://github.com/mj-will/nessai/pull/362))
- Add `log-posterior-weights` to `nessai.samplers.importance.ImportanceNestedSampler` (https://github.com/mj-will/nessai/pull/382)
- Add explicit support for Python 3.12 (https://github.com/mj-will/nessai/pull/374)
- Add fractional evidence stopping criterion to the importance nested sampler (https://github.com/mj-will/nessai/pull/371)
- Add option to recompute `log_q` when resuming the importance nested sampler instead of saving it (https://github.com/mj-will/nessai/pull/368)

### Changed

- Standardize how sampling history (run statistics) are stored ([#364](https://github.com/mj-will/nessai/pull/364))
- The importance nested sampler no longer requires the `to_unit_hypercube` method to run ([#362](https://github.com/mj-will/nessai/pull/362))
- The `ratio` stopping criterion is now computed using the log-likelihood threshold instead of the live points ([#362](https://github.com/mj-will/nessai/pull/362))
- Change various defaults related to the importance nested sampler ([#362](https://github.com/mj-will/nessai/pull/362))
- Random seed is now randomly set if not specified and saved in the result file (https://github.com/mj-will/nessai/pull/378)
- Rework how weights are handled in the importance nested sampler (https://github.com/mj-will/nessai/pull/376)

### Fixed

- Fix bug in with legend in `nessai.plot.plot_1d_comparison` ([#360](https://github.com/mj-will/nessai/pull/360))
- Fix bug with `truths` argument in `nessai.plot.corner_plot` (https://github.com/mj-will/nessai/pull/375)

### Removed

- Remove the deprecated `max_threads` argument from `nessai.flowsampler.FlowSampler` and `nessai.utils.threading.configure_threads` ([#363](https://github.com/mj-will/nessai/pull/363))

## [0.11.0]

### Added

- Add log-posterior weights to the result dictionary and file ([#341](https://github.com/mj-will/nessai/pull/341))
- Add support for checkpoint callbacks ([#355](https://github.com/mj-will/nessai/pull/355))

### Changed

- Explicitly support and test against Python 3.11 ([#352](https://github.com/mj-will/nessai/pull/352))


## [0.10.1]

### Fixed

- Relax tolerance used when checking if the log-prior is vectorised such that bilby priors are treated as vectorised ([#343](https://github.com/mj-will/nessai/pull/343))

## [0.10.0]

### Added

- `birth_log_likelihoods` to `NestedSampler` and  `logL_birth` to the result dictionary ([#318](https://github.com/mj-will/nessai/pull/318))
- Support for non-vectorised log-prior functions ([#330](https://github.com/mj-will/nessai/pull/330))
- Add the live points to the trace plot for the standard nested sampler ([#334](https://github.com/mj-will/nessai/pull/334))
- Add an option to resume from a pickle object rather than a resume file ([#337](https://github.com/mj-will/nessai/pull/337))

### Changed

- Nested samples are now stored as an array in the result object rather than a dictionary ([#318](https://github.com/mj-will/nessai/pull/318))
- Reduce the size of importance nested sampling checkpoints ([#327](https://github.com/mj-will/nessai/pull/327))
- Rename `nessai.utils.bilbyutils` to `nessai.utils.settings` ([#332](https://github.com/mj-will/nessai/pull/332))
- Changed name of `dZ` to `dlogZ`, this does not change how the stopping criterion is calculated ([#333](https://github.com/mj-will/nessai/pull/333))

### Fixed

- Fix a bug with the prior bounds that occurred when `bounds` and `names` had different orders ([#329](https://github.com/mj-will/nessai/pull/329))
- Fix a bug with `close_pool` that lead to the pool being closed irrespective of the value ([#331](https://github.com/mj-will/nessai/pull/331))

### Deprecated

- `nessai.utils.bilbyutils` is deprecated in favour on `nessai.utils.settings` and will be removed in a future release ([#332](https://github.com/mj-will/nessai/pull/332))

## [0.9.1]

### Fixed

- Fix duplicate parameters when adding reparameterisations (see [#320](https://github.com/mj-will/nessai/issues/320) for details) ([#321](https://github.com/mj-will/nessai/pull/321))

## [0.9.0]

### Added

- Add importance nested sampler ([#285](https://github.com/mj-will/nessai/pull/285))
- Add support for using regex for specifying parameters in the reparametersations dictionary ([#312](https://github.com/mj-will/nessai/pull/312))


### Changed

- Enable constant volume mode with uniform nball latent prior ([#306](https://github.com/mj-will/nessai/pull/306))
- Pass kwargs in RealNVP to the coupling class ([#307](https://github.com/mj-will/nessai/pull/307))
- Use log-scale on state plot ([#308](https://github.com/mj-will/nessai/pull/308))
- Support `forkserver` and `spawn` multiprocessing start methods ([#313](https://github.com/mj-will/nessai/pull/313))

### Fixed

- Fix resume bug with fallback reparameterisation ([#302](https://github.com/mj-will/nessai/pull/302))
- Fix bugs caused by numpy 1.25 ([#311](https://github.com/mj-will/nessai/pull/311))

## [0.8.1]

### Fixed

- Fix incorrect sign in delta phase reparameterisation ([#292](https://github.com/mj-will/nessai/pull/292))
- Remove maximum scipy version ([#295](https://github.com/mj-will/nessai/pull/295))
- Specify three quantiles in default corner kwargs as required by corner 2.2.2 ([#298](https://github.com/mj-will/nessai/pull/298))

## [0.8.0]

### Added

- Add `DeltaPhaseReparameterisation` for GW analyses. ([#244](https://github.com/mj-will/nessai/pull/244))
- Add `nessai.utils.sorting`. ([#244](https://github.com/mj-will/nessai/pull/244))
- Add `log_posterior_weights` and `effective_n_posterior_samples` to the integral state object. ([#248](https://github.com/mj-will/nessai/pull/248))
- Add a check for the multiprocessing start method when using `n_pool`. ([#250](https://github.com/mj-will/nessai/pull/250))
- Add option to reverse reparameterisations in `FlowProposal`.
- Add `disable_vectorisation` to `FlowSampler`. ([#254](https://github.com/mj-will/nessai/pull/254))
- Add `likelihood_chunksize` which allows the user to limit how many points are passed to a vectorised likelihood function at once. ([#256](https://github.com/mj-will/nessai/pull/256))
- Add `allow_multi_valued_likelihood` which allows for multi-valued likelihoods, e.g. that include numerical integration. ([#257](https://github.com/mj-will/nessai/pull/257))
- Add `parameters` keyword argument to `nessai.plot.plot_trace` and pass additional keyword arguments to the plotting function. ([#259](https://github.com/mj-will/nessai/pull/259))
- Add option to construct live points without non-sampling parameters. ([#266](https://github.com/mj-will/nessai/pull/266))
- Add option to use a different estimate of the shrinkage. Default remains unchanged. ([#248](https://github.com/mj-will/nessai/pull/248), [#269](https://github.com/mj-will/nessai/pull/269))
- Add `ScaleAndShift` reparameterisation which includes Z-score normalisation. ([#273](https://github.com/mj-will/nessai/pull/273))
- Add option to specify default result file extension. ([#274](https://github.com/mj-will/nessai/pull/274))

### Changed

- Refactor `nessai.reparameterisations` into a submodule. ([#241](https://github.com/mj-will/nessai/pull/241))
- Use `torch.inference_mode` instead of `torch.no_grad`. ([#245](https://github.com/mj-will/nessai/pull/245))
- Changed `CombinedReparameterisations` to sort and add reparameterisations based on their requirements. ([#244](https://github.com/mj-will/nessai/pull/244), [#253](https://github.com/mj-will/nessai/pull/253))
- Refactor `nessai.evidence._NSIntegralState` to inherit from a base class. ([#248](https://github.com/mj-will/nessai/pull/248))
- Revert default logging level to `INFO`. ([#249](https://github.com/mj-will/nessai/pull/249))
- Rework logging statements to reduce the amount of information printed by default. ([#249](https://github.com/mj-will/nessai/pull/249))
- Refactor `nessai.proposal.FlowProposal.verify_rescaling` to be stricter. ([#253](https://github.com/mj-will/nessai/pull/253))
- Truth input in `nessai.plot.corner_plot` can now be an iterable or a dictionary. ([#255](https://github.com/mj-will/nessai/pull/255))
- Tweak how the prior volume is computed for the final nested sample. This will also change the evidence and posterior weights. ([#248](https://github.com/mj-will/nessai/pull/248), [#269](https://github.com/mj-will/nessai/pull/269))
- Stricter handling of keyword arguments passed to `NestedSampler`. Unknown keyword arguments will now raise an error. ([#270](https://github.com/mj-will/nessai/pull/270))
- Rework `nessai.config` to have `config.livepoints` and `config.plot` which contain global settings. Some of the setting names have also changed. ([#272](https://github.com/mj-will/nessai/pull/272))
- `Rescale` reparameterisation is now an alias for `ScaleAndShift`. ([#273](https://github.com/mj-will/nessai/pull/273))
- Change the default result file extension to `hdf5`, old result file format can be recovered by setting it to `json`. ([#274](https://github.com/mj-will/nessai/pull/274))
- Optimisations to `FlowProposal.populate`, including changes to `Model.in_bounds` and how sampling from the latent prior is handled. ([#277](https://github.com/mj-will/nessai/pull/277))
- Add a maximum figure size (`nessai.config.plotting.max_figsize`) to prevent very large trace plots when the number of dimensions is very high. ([#282](https://github.com/mj-will/nessai/pull/282))

### Fixed

- Fix a bug where setting the livepoint precision (e.g. `f16`) did not work. ([#272](https://github.com/mj-will/nessai/pull/272))
- Fix plotting failing when sampling large number of parameters. ([#281](https://github.com/mj-will/nessai/pull/281), [#282](https://github.com/mj-will/nessai/pull/282))

### Removed

- Removed `nessai._NSIntegralState.reset`. ([#248](https://github.com/mj-will/nessai/pull/248))
- Removed `nessai.gw.legacy`. ([#267](https://github.com/mj-will/nessai/pull/267))
- Removed support for changing the variance of the latent distribution via `draw_latent_kwargs` from `FlowProposal`. ([#277](https://github.com/mj-will/nessai/pull/277))

## [0.7.1]

### Fixed

- Fix bug that led to the multiprocessing pool not being used when resuming. ([#261](https://github.com/mj-will/nessai/pull/261))

## [0.7.0]

**Important:** in this release the flow backend changed from `nflows` to `glasflow` which increased the minimum version of PyTorch to 1.11.0.

### Added

- Add explicit support for Python 3.10. ([#224](https://github.com/mj-will/nessai/pull/224))
- Add more structure utils (`get_subset_arrays`, `isfinite_struct`). ([#209](https://github.com/mj-will/nessai/pull/209))
- Add `nessai.sampler.base.BaseNestedSampler` class. ([#210](https://github.com/mj-will/nessai/pull/210))
- Add option to use multinomial resampling to `nessai.posterior.draw_posterior_samples`. ([#213](https://github.com/mj-will/nessai/pull/213), [#214](https://github.com/mj-will/nessai/pull/214))
- Add features (`log_prob`, `sample`, `end_iteration`, `finalise`, training with weights) to `FlowModel`. ([#216](https://github.com/mj-will/nessai/pull/216))
- Add option to checkpoint based on elapsed time. ([#225](https://github.com/mj-will/nessai/pull/225))
- Add `stream` option to `setup_logger` for setting the stream for `logging.StreamHandler`. ([#229](https://github.com/mj-will/nessai/pull/229))
- Add configurable periodic logging based on either the iteration or elapsed time. ([#229](https://github.com/mj-will/nessai/pull/229))
- Add `glasflow` dependency. ([#228](https://github.com/mj-will/nessai/pull/228))
- Add `posterior_sampling_method` to `FlowSampler.run`. ([#233](https://github.com/mj-will/nessai/pull/233))
- Add options `plot_{indices, posterior, logXlogL}` for disabling plots in `FlowSampler.run`. ([#233](https://github.com/mj-will/nessai/pull/233))
- Add `FlowSampler.terminate_run`. ([#233](https://github.com/mj-will/nessai/pull/233))
- Add `FlowSampler.log_evidence` and `FlowSampler.log_evidence_error`. ([#233](https://github.com/mj-will/nessai/pull/233))
- Add `nessai.utils.bilbyutils`. ([#236](https://github.com/mj-will/nessai/pull/236))
- Add a warning for when the final p-value for the insertion indices is less than 0.05. ([#235](https://github.com/mj-will/nessai/pull/235))
- Add `reset_flow` to `NestedSampler` for resetting the entire flow. ([#238](https://github.com/mj-will/nessai/pull/238))

### Changed

- Change how threading is handled to no longer use `max_threads`. ([#208](https://github.com/mj-will/nessai/pull/208))
- Refactor `nessai.nestedsampler` into the `nessai.samplers` submodule. ([#210](https://github.com/mj-will/nessai/pull/210))
- Refactor `nessai.flowmodel` into a submodule with `nessai.flowmodel.{base, utils, config}`. ([#216](https://github.com/mj-will/nessai/pull/216))
- Change how `noise_scale` is configured `FlowModel`. User can now specify `noise_type` and `noise_scale`. ([#216](https://github.com/mj-will/nessai/pull/216))
- Change `nessai.utils.rescaling.{logit, sigmoid}` to match `torch.{logit, sigmoid}`. ([#218](https://github.com/mj-will/nessai/pull/218))
- Change default checkpoint interval to 10 minutes rather than after training. ([#225](https://github.com/mj-will/nessai/pull/225))
- Change flows to use `glasflow.nflows` instead of `nflows`. ([#228](https://github.com/mj-will/nessai/pull/228))
- Change `close_pool` to be called at the end of `FlowSampler.run` rather than at the end of `NestedSampler.nested_sampling_loop`. ([#233](https://github.com/mj-will/nessai/pull/233))
- Bump minimum PyTorch version to 1.11.0. ([#230](https://github.com/mj-will/nessai/pull/230))

### Fixed

- Fixed a bug in `nessai.flows.utils.configure_model` that only occurred when the specified `device_tag` is invalid. ([#216](https://github.com/mj-will/nessai/pull/216))
- Fixed a bug in `nessai.utils.sampling.draw_truncated_gaussian` where the input was being changed by an in-place operation. ([#217](https://github.com/mj-will/nessai/pull/217))
- Fixed an infinite loop when resuming a run that was interrupted when switching proposal. ([#237](https://github.com/mj-will/nessai/pull/237))

### Deprecated

- Setting `max_threads` is deprecated and will be removed in a future release. ([#208](https://github.com/mj-will/nessai/pull/208))
- `nessai.nestedsampler` is deprecated and will be removed in a future release. Use `nessai.samplers.nestedsampler` instead. ([#226](https://github.com/mj-will/nessai/pull/226))
- `nessai.flows.transforms.LULinear` is deprecated in favour of `glasflow.nflows.transforms.LULinear` and will be removed in a future release. ([#228](https://github.com/mj-will/nessai/pull/228))

### Removed

- Removed unused code for saving live points in `NestedSampler`. ([#210](https://github.com/mj-will/nessai/pull/210))
- Removed `nflows` dependency. ([#228](https://github.com/mj-will/nessai/pull/228))

## [0.6.0] - 2022-08-24

### Added

- Add a warning in `Model.verify_model` when `Model.log_prior` returns an array that has `float16` precision. ([#175](https://github.com/mj-will/nessai/pull/175))
- Add more functionality for configuring live point fields and defaults. ([#170](https://github.com/mj-will/nessai/pull/170))
- Record iteration at which live points are drawn in `it` field of live points. ([#170](https://github.com/mj-will/nessai/pull/170))
- Add `nessai.config` for storing package wide defaults. ([#170](https://github.com/mj-will/nessai/pull/170))
- Add `nessai.utils.testing` submodule which contains functions to use during testing. ([#170](https://github.com/mj-will/nessai/pull/170))
- Add `nessai.livepoint.unstructured_view` and `nessai.model.Model.unstructured_view` for constructing unstructured views of live points. ([#178](https://github.com/mj-will/nessai/pull/178))
- Add `nessai.plot.corner_plot` as an alternative to `plot_live_points` that uses `corner` instead of `seaborn`. ([#189](https://github.com/mj-will/nessai/pull/189))
- Add new examples. ([#195](https://github.com/mj-will/nessai/pull/195), [#198](https://github.com/mj-will/nessai/pull/198))
- Add `filehandler_kwargs` to `nessai.utils.logging.setup_logger` which allows the user to configure the `FileHandler` in the logger. ([#204](https://github.com/mj-will/nessai/pull/204))
- Add `final_p_value` and `final_ks_statistic` to `NestedSampler` and the result file.

### Changed

- Change default values for log-likelihood and log-prior in empty live points to be `np.nan` instead of zero. ([#170](https://github.com/mj-will/nessai/pull/170))
- `nessai.livepoint.get_dtype` now returns an instance of `numpy.dtype`. ([#170](https://github.com/mj-will/nessai/pull/170))
- Style for plots is no longer set globally and can be disabled completely. ([#194](https://github.com/mj-will/nessai/pull/194))
- Update examples. ([#190](https://github.com/mj-will/nessai/pull/190))
- Changed behaviour of `from nessai import *` to no longer imports any modules. ([#201](https://github.com/mj-will/nessai/pull/201))

### Fixed

- Fixed a bug in `FlowProposal.populate` which occurred when the pool of samples was not empty (closes [#176](https://github.com/mj-will/nessai/issues/176)) ([#177](https://github.com/mj-will/nessai/pull/177))
- Fixed a bug in `nessai.model.Model.new_point` where the incorrect number of points were returned. ([#200](https://github.com/mj-will/nessai/pull/200))

### Removed

- Drop support for Python 3.6. ([#188](https://github.com/mj-will/nessai/pull/188))
- Remove a temporary fix for [#46](https://github.com/mj-will/nessai/issues/46) that was introduced in [#47](https://github.com/mj-will/nessai/pull/47). ([#202](https://github.com/mj-will/nessai/pull/202))

## [0.5.1] - 2022-06-20

### Fixed

- Fixed a bug where live points where added to the initial points with incorrect log-likelihood and log-prior. ([#171](https://github.com/mj-will/nessai/pull/171))

## [0.5.0] - 2022-06-14

### Added

- Add `dataframe_to_live_points` function to `nessai.livepoint` for converting from a `pandas.DataFrame` to live points. ([#133](https://github.com/mj-will/nessai/pull/133))
- Add `fallback_reparameterisation` to `FlowProposal`. This allows the user to specify which reparameterisation to use for parameters that are not included in the reparameterisations dictionary. Default behaviour remains unchanged (defaults to no reparameterisation). ([#134](https://github.com/mj-will/nessai/pull/134))
- Add `rolling_mean` to `nessai.utils.stats`. ([#136](https://github.com/mj-will/nessai/pull/136))
- Add `nessai.flows.utils.create_linear_transform` as a common function for creating linear transforms in the flows. ([#137](https://github.com/mj-will/nessai/pull/137))
- Add `nessai.flows.transforms.LULinear` to address a [bug in nflows](https://github.com/bayesiains/nflows/pull/38) that has not been patched and prevents the use of CUDA with `LULinear`. ([#138](https://github.com/mj-will/nessai/pull/138))
- Add `calibration_example.py` to the gravitational wave examples. ([#139](https://github.com/mj-will/nessai/pull/139))
- Add `defaults` keyword argument to `nessai.reparameterisations.get_reparameterisation` for overriding the dictionary of default reparameterisations. ([#142](https://github.com/mj-will/nessai/pull/142))
- Add explicit tests for `nessai.flowsampler` ([#143](https://github.com/mj-will/nessai/pull/143))
- Add more tests for `nessai.reparameterisations` ([#145](https://github.com/mj-will/nessai/pull/145))
- Add more tests for `nessai.gw` ([#144](https://github.com/mj-will/nessai/pull/144))
- Add support for vectorised likelihoods and automatically detect if the likelihood is vectorised. ([#148](https://github.com/mj-will/nessai/pull/148), [#166](https://github.com/mj-will/nessai/pull/166))
- Add support for passing a user-defined pool instead of using `n_pool`. ([#148](https://github.com/mj-will/nessai/pull/148))
- Add an option to disable signal handling in `FlowSampler`. ([#159](https://github.com/mj-will/nessai/pull/159))
- Add support for `ray.util.multiprocessing.Pool` (fixes [#162](https://github.com/mj-will/nessai/issues/162)). ([#163](https://github.com/mj-will/nessai/pull/163))

### Changed

- `NestedSampler.plot_state` now includes the log-prior volume in one of the subplots and the rolling mean of the gradient (|dlogL/dLogX|) is plotted instead of the gradient directly. ([#136](https://github.com/mj-will/nessai/pull/136))
- The figure produced by `NestedSampler.plot_state` now includes a legend for the different vertical lines that can appear in the subplots. ([#136](https://github.com/mj-will/nessai/pull/136))
- `RealNVP` and `NeuralSplineFlow` now use `nessai.flows.utils.create_linear_transform`. ([#137](https://github.com/mj-will/nessai/pull/137))
- Updated all of the examples to reflect the new defaults. ([#139](https://github.com/mj-will/nessai/pull/139))
- Rework `nessai.gw.reparameterisations.get_gw_reparameterisation` to use `get_reparameterisation` with the `defaults` keyword argument. ([#142](https://github.com/mj-will/nessai/pull/142))
- Switch to `os.path.join` for joining paths. ([#143](https://github.com/mj-will/nessai/pull/143), [#161](https://github.com/mj-will/nessai/pull/161))
- Context is now passed to the transform in `nessai.flows.base.NFlow` enabling the use of flows with conditional transforms. ([#146](https://github.com/mj-will/nessai/pull/146))
- Add `context_features` to RealNVP and NeuralSplineFlows ([#146](https://github.com/mj-will/nessai/pull/146))
- Rework `MaskedAutoregressiveFlow` to add `context_features` ([#146](https://github.com/mj-will/nessai/pull/146))
- Rework how likelihood parallelisation is handled. The model now contains the pool instead of the sampler and proposals. ([#148](https://github.com/mj-will/nessai/pull/148))
- Update `parallelisation_example.py` to show use of `n_pool` and `pool` for parallelisation. ([#148](https://github.com/mj-will/nessai/pull/148))
- Simplify how the normalising flow is reset in `FlowModel` and `NestedSampler`. ([#150](https://github.com/mj-will/nessai/pull/150))
- Reduce logging level a some statements in `FlowProposal`. ([#160](https://github.com/mj-will/nessai/pull/160))


### Fixed

- Fixed a bug in `RescaleToBounds` when using `pre_rescaling` without boundary inversion. ([#145](https://github.com/mj-will/nessai/pull/145))
- Fixed slow integration tests not running if a quick integration test is reran after failing. ([#153](https://github.com/mj-will/nessai/pull/153))
- Fixed a bug that prevented the use of `prior_sampling=True` with `FlowSampler`. ([#156](https://github.com/mj-will/nessai/pull/156))
- Fix issue when creating multiple instances of `FlowSampler` with the same output directory when resuming is enabled as reported in [#155](https://github.com/mj-will/nessai/issues/155). ([#157](https://github.com/mj-will/nessai/pull/157))
- Fixed missing square-root in `nessai.flows.distributions.MultivariateGaussian._sample` and fix the corresponding unit test. ([#158](https://github.com/mj-will/nessai/pull/158))
- Fix issue with cosmology in `ComovingDistanceConverter` caused by changes to `astropy.cosmology` in version 5.1. ([#168](https://github.com/mj-will/nessai/pull/168))
- Fixed bug with caching in `LULinear` transform when reloading a checkpointed flow. ([#167](https://github.com/mj-will/nessai/pull/167))


### Removed

- Removed `legacy_gw_example.py` ([#139](https://github.com/mj-will/nessai/pull/139))
- Removed `keep_samples` from `FlowProposal`. ([#140](https://github.com/mj-will/nessai/pull/140))

## [0.4.0] - 2021-11-23

### Added

- Add a constant volume mode to `FlowProposal`. In this mode the radius of the latent contour is fixed to the q'th quantile, which by default is `0.95`. ([#125](https://github.com/mj-will/nessai/pull/125))
- Add a check for `resume_file` when `resume=True`. ([#126](https://github.com/mj-will/nessai/pull/126))
- Change default logging level to `WARNING`. ([#126](https://github.com/mj-will/nessai/pull/126))
- Add `angle-cosine` reparameterisation. ([#126](https://github.com/mj-will/nessai/pull/126))
- Added an explicit check for one-dimensional models that raises a custom exception `OneDimensionalModelError`. ([#121](https://github.com/mj-will/nessai/pull/121))
- `RealNVP` and `NeuralSplineFlow` now raise an error if `features<=1`. ([#121](https://github.com/mj-will/nessai/pull/121))
- Add option in `nessai.reparameterisations.Angle` to set `scale=None`, the scale is then set as `2 * pi / angle_prior_range`. ([#127](https://github.com/mj-will/nessai/pull/127))
- Add `'periodic'` reparameterisation that uses `scale=None` in `nessai.reparameterisations.Angle`. ([#127](https://github.com/mj-will/nessai/pull/127))
- Add the `use_default_reparameterisations` option to `FlowProposal` to allow the use of the default reparameterisations in `GWFlowProposal` without specifying any reparameterisations. ([#129](https://github.com/mj-will/nessai/pull/129))
- Add `chi_1`, `chi_2` and `time_jitter` to known parameters in `GWFlowProposal` with corresponding defaults. ([#130](https://github.com/mj-will/nessai/pull/130))

### Changed

- Reparameterisation `angle-sine` is now an alias for `RescaledToBounds` instead of `Angle` with specific keyword arguments. ([#126](https://github.com/mj-will/nessai/pull/126))
- `maximum_uninformed=None` now defaults to 2 times `nlive` instead of `np.inf`. ([#126](https://github.com/mj-will/nessai/pull/126))
- `nlive=2000` by default. ([#126](https://github.com/mj-will/nessai/pull/126))
- Default `batch_size` is now 1000. ([#126](https://github.com/mj-will/nessai/pull/126))
- Default `n_neurons` is now 2 times the dimensions of the normalising flow. ([#126](https://github.com/mj-will/nessai/pull/126))
- Default mode for `FlowProposal` is `constant_volume_mode=True`. ([#126](https://github.com/mj-will/nessai/pull/126))
- Proposal plots are now disabled by default. ([#126](https://github.com/mj-will/nessai/pull/126))
- `cooldown` now defaults to `200` to reflect the change in `nlive`. ([#126](https://github.com/mj-will/nessai/pull/126))
- Default optimiser is now `adamw`. ([#126](https://github.com/mj-will/nessai/pull/126))
- Rework `AugmentedFlowProposal` to work with the new defaults. ([#126](https://github.com/mj-will/nessai/pull/126))
- `Model.names` and `Model.bounds` are now properties by default and their setters include checks to verify the values provided are valid and raise errors if not. ([#121](https://github.com/mj-will/nessai/pull/121))
- Logger now has propagation enabled by default. ([#128](https://github.com/mj-will/nessai/pull/128))
- `FlowProposal.configure_reparameterisations` can now handle an input of `None`. In this case only the default reparameterisations will be added. ([#129](https://github.com/mj-will/nessai/pull/129))
- Changed default reparameterisation for gravitational-wave parameters `a_1` and `a_2` to `'default'`. ([#130](https://github.com/mj-will/nessai/pull/130))

### Fixed

- Fixed a bug where the parameters list passed to `Reparameterisation` (or its child classes) wasn't being copied and changes made within the reparameterisation would change the original list. ([#127](https://github.com/mj-will/nessai/pull/127))

### Deprecated

- `keep_samples` in `FlowProposal` will be removed in the next minor release.


## [0.3.3] - 2021-11-04

### Fixed

- Fixed a bug in `nessai.livepoint.dict_to_live_points` when passing a dictionary where the entries contained floats instead of objects with a length raised an error. ([#119](https://github.com/mj-will/nessai/pull/119))


## [0.3.2] - 2021-10-12

### Added

- Added more checks to the init method for `nessai.reparameterisations.AnglePair` to catch invalid combinations of priors and/or angle conventions. Now supports RA or azimuth defined on [-pi, pi] in addition to [0, 2pi]. ([#114](https://github.com/mj-will/nessai/pull/114))
- Add a check in `nessai.flowmodel.update_config` for `'noise_scale'`, a `ValueError` is now raised if `noise_scale` is not a float or `'adaptive'`. ([#115](https://github.com/mj-will/nessai/pull/115))
- Add `codespell` to the pre-commit checks. ([#116](https://github.com/mj-will/nessai/pull/116))

### Changed

- The dtype for tensors passed to the flow is now set using `torch.get_default_dtype()` rather than always using `float32`. ([#108](https://github.com/mj-will/nessai/pull/108))
- Incorrect values for `mask` in `nessai.flows.realnvp.RealNVP` now raise `ValueError` and improved the error messages returned by all the exceptions in the class. ([#109](https://github.com/mj-will/nessai/pull/109))
- Change scale of y-axis of the log-prior volume vs. log-likelihood plot from `symlog` to the default linear axis. ([#110](https://github.com/mj-will/nessai/pull/110))
- `nessai.plot.plot_trace` now includes additional parameters such as `logL` and `logP` default, previously the last two parameters (assumed to be `logL` and `logP` were always excluded). ([#111](https://github.com/mj-will/nessai/pull/111))

### Fixed

- Fixed an issue where `nessai.reparameterisations.AnglePair` would silently break when the prior range for RA or azimuth was set to a range that wasn't [0, 2pi]. It now correctly handles both [0, 2pi] and [-pi, pi] and raises an error for any other ranges. ([#114](https://github.com/mj-will/nessai/pull/114))
- Fixed various spelling mistakes throughout the source code and documentation. ([#116](https://github.com/mj-will/nessai/pull/116))


## [0.3.1] Minor improvements and bug fixes - 2021-08-23

This release has a few minor improvements and bug fixes. It also explicitly adds support for python 3.9, which worked previously but was not tested.
### Added

- Add `in_bounds`, `parameter_in_bounds` and `sample_parameter` methods to `nessai.model.Model`. ([#90](https://github.com/mj-will/nessai/pull/90))
- Implemented the option to specify the cosmology in `nessai.gw.utils.ComovingDistanceConverter` using `astropy`. Previously changing the value had no effect of the transformation. ([#91](https://github.com/mj-will/nessai/pull/91))
- Improve test coverage for `nessai.proposal.base.Proposal` ([#92](https://github.com/mj-will/nessai/pull/92))
- Add `'logit'` to the default reparameterisations ([#98](https://github.com/mj-will/nessai/pull/98))
- Add example using the Rosenbrock likelihood in two dimensions ([#99](https://github.com/mj-will/nessai/pull/99))
- Add a `colours` argument to `nessai.plot.plot_1d_comparison` ([#102](https://github.com/mj-will/nessai/pull/102))
- Explicitly support Python 3.9 (Added Python 3.9 to unit tests) ([#103](https://github.com/mj-will/nessai/pull/103))

### Changed

- `nessai.gw.utils.DistanceConverter` now inherits from `abc.ABC` and `to_uniform_parameter` and `from_uniform_parameter` are both abstract methods. ([#91](https://github.com/mj-will/nessai/pull/91))
- `nessai.proposal.base.Proposal` now inherits from `abc.ABC` and `draw` is an abstract method. ([#92](https://github.com/mj-will/nessai/pull/92))
- `nessai.proposal.rejection.RejectionProposal` now inherits from `nessai.proposal.analytic.AnalyticProposal`. Functionality is the same but the code will be easier to maintain since this removes several methods that were identical. ([#93](https://github.com/mj-will/nessai/pull/93))
- `noise_scale='adaptive'` option in `FlowModel` now correctly uses a standard deviation of 0.2 times the mean nearest neighbour separation as described in [Moss 2019](https://arxiv.org/abs/1903.10860). Note that this feature is disabled by default, so this does not change the default behaviour. ([#95](https://github.com/mj-will/nessai/pull/95))
- Refactor `nessai.utils` into a submodule. ([#96](https://github.com/mj-will/nessai/pull/96))
- Change behaviour of `determine_rescaled_bounds` so that `rescale_bounds` is ignored when `inversion=True`. This matches the behaviour in `RescaledToBounds` where when boundary inversion is enabled, values are rescaled to [0, 1] and then if no inversion if applied, changed to [-1, 1]. ([#96](https://github.com/mj-will/nessai/pull/96))
- Tweaked `detect_edges` so that `both` is returned in cases where the lower and upper regions contain zero probability. ([#96](https://github.com/mj-will/nessai/pull/96))
- `NestedSampler` no longer checks capitalisation of `flow_class` when determining which proposal class to use. E.g. `'FlowProposal'` and `'flowproposal'` are now both valid values. ([#100](https://github.com/mj-will/nessai/pull/100))
- `NestedSampler.configure_flow_proposal` now raises `ValueError` instead of `RuntimeError` if `flow_class` is an invalid string. ([#100](https://github.com/mj-will/nessai/pull/100))
- Raise a `ValueError` if `nessai.plot.plot_1d_comparison` is called with a labels list and the length does not match the number of sets of live points being compared. ([#102](https://github.com/mj-will/nessai/pull/102))
- `nessai.flow.base.BaseFlow` now also inherits from `abc.ABC` and methods that should be defined by the user are abstract methods. ([#104](https://github.com/mj-will/nessai/pull/104))
- Changed default to `fuzz=1e-12` in `nessai.utils.rescaling.logit` and `nessai.utils.rescaling.sigmoid` and improved stability. ([#105](https://github.com/mj-will/nessai/pull/105))

### Fixed

- Fixed a typo in `nessai.gw.utils.NullDistanceConverter.from_uniform_parameter` that broke the method. ([#91](https://github.com/mj-will/nessai/pull/91))
- Fixed a bug in `nessai.reparameterisations.RescaleToBounds` when using `offset=True` and `pre_rescaling` where the prime prior bounds were incorrectly set. ([#97](https://github.com/mj-will/nessai/pull/97))
- Fixed a bug that prevented disabling periodic checkpointing. ([#101](https://github.com/mj-will/nessai/pull/101))
- Fixed a bug when calling `nessai.plot.plot_1d_comparison` with live points that contain a field with only infinite values. ([#102](https://github.com/mj-will/nessai/pull/102))
- Fixed the log Jacobian determinant for `nessai.utils.rescaling.logit` and `nessai.utils.rescaling.sigmoid` which previously did not include the Jacobian for the fuzz when it was used. ([#105](https://github.com/mj-will/nessai/pull/105))


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
- Explicitly check prior bounds when using reparameterisations. This catches cases where infinite bounds are used and break some reparameterisations. (#82)
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
- Plotting functions are now more consistent and all return the figure if `filename=None`.

### Fixed

- Fixed bug when plotting non-structured arrays with `plot_1d_comparison` and specifying `parameters`.
- Fixed bug where `plot_indices` failed if using an empty array but worked with an empty list.

### Removed

- Remove `plot_posterior` because functionality is include in `plot_live_points`.
- Remove `plot_likelihood_evaluations` because information is already contained in the state plot.
- Remove `plot_acceptance` as it is only by augmented proposal which is subject to change.
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

[Unreleased]: https://github.com/mj-will/nessai/compare/v0.13.2...HEAD
[0.13.2]: https://github.com/mj-will/nessai/compare/v0.13.1...v0.13.2
[0.13.1]: https://github.com/mj-will/nessai/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/mj-will/nessai/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/mj-will/nessai/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/mj-will/nessai/compare/v0.10.1...v0.11.0
[0.10.1]: https://github.com/mj-will/nessai/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/mj-will/nessai/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/mj-will/nessai/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/mj-will/nessai/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/mj-will/nessai/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/mj-will/nessai/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/mj-will/nessai/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/mj-will/nessai/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/mj-will/nessai/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/mj-will/nessai/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/mj-will/nessai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/mj-will/nessai/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/mj-will/nessai/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/mj-will/nessai/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/mj-will/nessai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/mj-will/nessai/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/mj-will/nessai/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/mj-will/nessai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/mj-will/nessai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/mj-will/nessai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mj-will/nessai/compare/v0.1.1...v0.2.0
