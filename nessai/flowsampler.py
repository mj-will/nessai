# -*- coding: utf-8 -*-
"""
Main code that handles running and checkpoiting the sampler.
"""
import logging
import os
import signal
import sys
from typing import Optional

from . import config
from .livepoint import live_points_to_dict
from .samplers import NestedSampler, ImportanceNestedSampler
from .posterior import draw_posterior_samples
from .utils import configure_threads
from .utils.torchutils import set_torch_default_dtype
from .utils.io import save_to_json, save_dict_to_hdf5


logger = logging.getLogger(__name__)


class FlowSampler:
    """
    Main class to handle running the nested sampler.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User-defined model.
    output : str, optional
        Output directory
    resume : bool, optional
        If True try to resume the sampler is the resume file exists.
    resume_file : str, optional
        File to resume sampler from.
    resume_data: Any, optional
        Data to resume the sampler from instead of a resume file. The data will
        be passed to the :code:`resume_from_pickled_sampler` of the relevant
        class.
    weights_path : str, optional
        Path to either the weights file or directory containing subdirectories
        with weight files.
    weights_file : str, optional
        Weights file to use, same as :code:`weights_path` and included for
        backwards_compatibility. Will be ignored is :code:`weights_paths` has
        also been specified.
    pytorch_threads : int
        Maximum number of threads to use for torch. If ``None`` torch uses all
        available threads.
    signal_handling : bool
        Enable or disable signal handling.
    exit_code : int, optional
        Exit code to use when forcibly exiting the sampler.
    close_pool : bool
        If True, the multiprocessing pool will be closed once the run method
        has been called. Disables the option in :code:`NestedSampler` if
        enabled.
    disable_vectorisation : bool
        Disable likelihood vectorisation. Overrides the value of
        :py:attr:`nessai.model.Model.allow_vectorised`.
    likelihood_chunksize : Optional[int]
        Chunksize used when evaluating a vectorised likelihood. Overrides the
        of :py:attr:`nessai.model.Model.likelihood_chunksize`. Set to None to
        evaluate the likelihood with all available points.
    allow_multi_valued_likelihood : Optional[bool]
        Allow for a multi-valued likelihood function that will return different
        likelihood values for the same point in parameter space. See
        :py:attr:`nessai.model.Model.allow_multi_valued_likelihood` for more
        details.
    parallelise_prior : Optional[bool]
        If true, and a multiprocessing pool has been specified, then the
        log-prior calculation will be parallelised using multiprocessing.
        If false, then the pool with not be used. Overrides the value of the
        `parallelise_prior` attribute in the model class, which is false by
        default.
    result_extension : str
        Extension used when saving the result format. Defaults to HDF5, but
        also supports JSON.
    kwargs :
        Keyword arguments passed to
        :obj:`~nessai.samplers.nestedsampler.NestedSampler`.
    """

    def __init__(
        self,
        model,
        output=os.getcwd(),
        importance_nested_sampler=False,
        resume=True,
        resume_file="nested_sampler_resume.pkl",
        resume_data=None,
        weights_file=None,
        weights_path=None,
        signal_handling=True,
        exit_code=130,
        pytorch_threads=1,
        close_pool=True,
        eps=None,
        torch_dtype=None,
        disable_vectorisation=False,
        likelihood_chunksize=None,
        allow_multi_valued_likelihood=None,
        parallelise_prior=None,
        result_extension="hdf5",
        **kwargs,
    ):

        configure_threads(
            pytorch_threads=pytorch_threads,
        )

        self._final_samples = None
        self._nested_samples = None
        self.exit_code = exit_code
        self.eps = eps
        if self.eps is not None:
            logger.info(f"Setting eps to {self.eps}")
            config.general.eps = self.eps

        self.torch_dtype = set_torch_default_dtype(torch_dtype)

        if importance_nested_sampler:
            SamplerClass = ImportanceNestedSampler
        else:
            SamplerClass = NestedSampler
        self.importance_nested_sampler = importance_nested_sampler

        self.close_pool = close_pool

        self.result_extension = result_extension

        if disable_vectorisation:
            logger.warning(
                "Overriding value of `allow_vectorised` in the model"
            )
            model.allow_vectorised = False

        if likelihood_chunksize:
            model.likelihood_chunksize = likelihood_chunksize

        if allow_multi_valued_likelihood is not None:
            model.allow_multi_valued_likelihood = allow_multi_valued_likelihood

        if parallelise_prior is not None:
            logger.debug("Overriding `parallelise_prior` in the model")
            model.parallelise_prior = parallelise_prior

        self.output = os.path.join(output, "")
        os.makedirs(self.output, exist_ok=True)
        self.save_kwargs(kwargs)

        model.configure_pool(
            n_pool=kwargs.pop("n_pool", None), pool=kwargs.pop("pool", None)
        )

        if resume and self.check_resume(resume_file, resume_data):
            weights_path = weights_path or weights_file
            if resume_data is not None:
                self.ns = self._resume_from_data(
                    SamplerClass,
                    resume_data=resume_data,
                    model=model,
                    weights_path=weights_path,
                    flow_config=kwargs.get("flow_config"),
                    checkpoint_callback=kwargs.get("checkpoint_callback"),
                )
            else:
                self.ns = self._resume_from_file(
                    SamplerClass,
                    model=model,
                    resume_file=resume_file,
                    weights_path=weights_path,
                    flow_config=kwargs.get("flow_config"),
                    checkpoint_callback=kwargs.get("checkpoint_callback"),
                )
        else:
            logger.debug("Not resuming sampler")
            self.ns = SamplerClass(
                model,
                output=self.output,
                resume_file=resume_file,
                close_pool=False,
                **kwargs,
            )

        if signal_handling:
            try:
                signal.signal(signal.SIGTERM, self.safe_exit)
                signal.signal(signal.SIGINT, self.safe_exit)
                signal.signal(signal.SIGALRM, self.safe_exit)
            except AttributeError:
                logger.error("Cannot set signal attributes on this system")
        else:
            logger.warning(
                "Signal handling is disabled. nessai will not automatically "
                "checkpoint when exited."
            )

    def check_resume(self, resume_file, resume_data):
        """Check if it is possible to resume the sampler"""
        return (
            resume_file
            and any(
                (
                    os.path.exists(os.path.join(self.output, f))
                    for f in [resume_file, resume_file + ".old"]
                )
            )
        ) or resume_data is not None

    def _resume_from_file(
        self,
        SamplerClass,
        resume_file,
        model,
        weights_path,
        flow_config,
        **kwargs,
    ):
        logger.info(f"Trying to resume sampler from {resume_file}")
        try:
            ns = SamplerClass.resume(
                os.path.join(self.output, resume_file),
                model,
                weights_path=weights_path,
                flow_config=flow_config,
                **kwargs,
            )
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(
                f"Could not load resume file from: {self.output} "
                f"with error {e}"
            )
            try:
                resume_file += ".old"
                ns = SamplerClass.resume(
                    os.path.join(self.output, resume_file),
                    model,
                    weights_path=weights_path,
                    flow_config=flow_config,
                    **kwargs,
                )
            except RuntimeError as e:
                logger.error(
                    "Could not load old resume " f"file from: {self.output}"
                )
                raise RuntimeError(
                    "Could not resume sampler " f"with error: {e}"
                )
        return ns

    def _resume_from_data(
        self,
        SamplerClass,
        resume_data,
        model,
        weights_path,
        flow_config,
        **kwargs,
    ):
        logger.info("Trying to resume sampler from `resume_data`")
        return SamplerClass.resume_from_pickled_sampler(
            resume_data,
            model,
            weights_path=weights_path,
            flow_config=flow_config,
            **kwargs,
        )

    @property
    def log_evidence(self):
        """Return the most recent log evidence"""
        return self.logZ

    @property
    def log_evidence_error(self):
        """Return the most recent log evidence error"""
        return self.logZ_error

    @property
    def nested_samples(self):
        """Return the nested samples"""
        if self._final_samples is not None:
            return self._final_samples
        else:
            return self._nested_samples

    def run(
        self,
        plot=True,
        save=True,
        posterior_sampling_method=None,
        close_pool=None,
        **kwargs,
    ):
        """Run the nested sampler.

        Will pick the correct run method given the configuration used.

        Parameters
        ----------
        plot : bool
            Toggle all plots produced once the sampler has converged.
        save : bool, optional
            Toggle automatic saving of results
        posterior_sampling_method : str, optional
            Method used for drawing posterior samples. Defaults to rejection
            sampling.
        close_pool : bool, optional
            Boolean to indicated if the pool should be closed at the end of the
            run function. If False, the user must manually close the pool. If
            specified, this value overrides the value passed when initialising
            the FlowSampler class.
        """
        if self.importance_nested_sampler:
            self.run_importance_nested_sampler(
                plot=plot,
                save=save,
                posterior_sampling_method=posterior_sampling_method,
                close_pool=close_pool,
                **kwargs,
            )
        else:
            self.run_standard_sampler(
                plot=plot,
                save=save,
                posterior_sampling_method=posterior_sampling_method,
                close_pool=close_pool,
                **kwargs,
            )

    def run_standard_sampler(
        self,
        plot=True,
        plot_indices=True,
        plot_posterior=True,
        plot_logXlogL=True,
        save=True,
        posterior_sampling_method=None,
        close_pool=None,
    ):
        """Run the standard nested sampler.

        Parameters
        ----------
        plot : bool
            Toggle all plots produced once the sampler has converged.
        plot_indices : bool
            Toggle the insertion indices plot.
        plot_posterior : bool
            Toggle the posterior distribution plot.
        plot_logXlogL : bool
            Toggle the log-prior volume vs log-likelihood plot.
        save
            Enable or disable saving of a results file.
        posterior_sampling_method
            Method used for drawing posterior samples. Defaults to rejection
            sampling.
        close_pool : bool
            Boolean to indicated if the pool should be closed at the end of the
            run function. If False, the user must manually close the pool. If
            specified, this value overrides the value passed when initialising
            the class.
        """
        if self.importance_nested_sampler:
            raise RuntimeError(
                "Cannot run standard sampler when importance_sampler=True"
            )
        if close_pool is None:
            close_pool = self.close_pool
        if posterior_sampling_method is None:
            posterior_sampling_method = "rejection_sampling"

        self.ns.initialise()
        self.logZ, self._nested_samples = self.ns.nested_sampling_loop()
        self.logZ_error = self.ns.state.log_evidence_error
        logger.info(f"Total sampling time: {self.ns.sampling_time}")
        logger.info(
            "Total likelihood evaluations: "
            f"{self.ns.total_likelihood_evaluations}"
        )

        logger.info("Starting post processing")
        logger.info("Computing posterior samples")
        self.posterior_samples = draw_posterior_samples(
            self._nested_samples,
            log_w=self.ns.state.log_posterior_weights,
            method=posterior_sampling_method,
        )
        logger.info(
            f"Returned {self.posterior_samples.size} " "posterior samples"
        )

        if save:
            self.save_results(
                os.path.join(self.output, "result"),
                extension=self.result_extension,
            )

        if plot:
            from nessai import plot

            if plot_posterior:
                plot.plot_live_points(
                    self.posterior_samples,
                    filename=os.path.join(
                        self.output, "posterior_distribution.png"
                    ),
                )
            if plot_indices:
                plot.plot_indices(
                    self.ns.insertion_indices,
                    self.ns.nlive,
                    filename=os.path.join(
                        self.output, "insertion_indices.png"
                    ),
                )

            if plot_logXlogL:
                self.ns.state.plot(os.path.join(self.output, "logXlogL.png"))

        if close_pool:
            self.ns.close_pool()

    def run_importance_nested_sampler(
        self,
        plot=True,
        plot_posterior=True,
        save=True,
        posterior_sampling_method=None,
        redraw_samples=False,
        n_posterior_samples=None,
        compute_initial_posterior=False,
        close_pool=None,
        **kwargs,
    ):
        """Run the importance nested sampler.

        Parameters
        ----------
        plot
            Enable or disable plotting. Independent of the value passed
            to the :code:`NestedSampler` object.
        save
            Enable or disable saving of a results file.
        posterior_sampling_method
            Method used for drawing posterior samples. Defaults to importance
            sampling.
        redraw_samples
            If True after the sampling is finished, samples are redrawn from
            the meta proposal and used to compute an updated evidence estimate
            and posterior. This can reduce biases in the results.
        n_posterior_samples
            Number of posterior samples to draw when when redrawing samples.
        compute_initial_posterior
            Enables or disables computing the posterior before redrawing
            samples. If :code:`redraw_samples` is False, then this flag is
            ignored.
        close_pool : bool
            Boolean to indicated if the pool should be closed at the end of the
            run function. If False, the user must manually close the pool. If
            specified, this value overrides the value passed when initialising
            the class.
        kwargs
            Keyword arguments passed to \
                :py:meth:`~nessai.importancesampler.ImportanceNestedSampler.draw_final_samples`
        """
        if not self.importance_nested_sampler:
            raise RuntimeError(
                "Cannot run importance sampler when importance_sampler=False"
            )
        if close_pool is None:
            close_pool = self.close_pool
        if posterior_sampling_method is None:
            posterior_sampling_method = "importance_sampling"

        self.ns.nested_sampling_loop()
        self._nested_samples = self.ns.samples
        self.logZ = self.ns.log_evidence
        self.logZ_error = self.ns.log_evidence_error
        logger.info(f"Total sampling time: {self.ns.sampling_time}")
        logger.info(
            "Total likelihood evaluations: "
            f"{self.ns.total_likelihood_evaluations}"
        )

        logger.info("Starting post processing")

        if redraw_samples:
            logger.info("Redrawing samples")
            self.initial_logZ = self.logZ
            self.initial_logZ_error = self.logZ_error
            _, self._final_samples = self.ns.draw_final_samples(
                n_post=n_posterior_samples,
                **kwargs,
            )
            self.logZ = self.ns.final_log_evidence
            self.logZ_error = self.ns.final_log_evidence_error

        logger.info("Computing posterior samples")

        if compute_initial_posterior or not redraw_samples:
            logger.debug("Computing posterior samples")
            self.initial_posterior_samples = self.ns.draw_posterior_samples(
                sampling_method=posterior_sampling_method,
                use_final_samples=False,
            )
        if redraw_samples:
            self.posterior_samples = self.ns.draw_posterior_samples(
                sampling_method=posterior_sampling_method,
                use_final_samples=True,
            )
        else:
            self.posterior_samples = self.initial_posterior_samples
        logger.info(
            f"Returned {self.posterior_samples.size} posterior samples"
        )

        if save:
            self.save_results(
                os.path.join(self.output, "result"),
                extension=self.result_extension,
            )

        if plot and plot_posterior:
            logger.debug("Producing plots")
            from nessai import plot

            plot.plot_live_points(
                self.posterior_samples,
                filename=os.path.join(
                    self.output, "posterior_distribution.png"
                ),
            )
            if redraw_samples and compute_initial_posterior:
                plot.plot_live_points(
                    self.initial_posterior_samples,
                    filename=os.path.join(
                        self.output, "initial_posterior_distribution.png"
                    ),
                )
        if close_pool:
            self.ns.close_pool()

    def save_kwargs(self, kwargs: dict) -> None:
        """
        Save the dictionary of keyword arguments used.

        Uses an encoder class to handle numpy arrays.

        Parameters
        ----------
        kwargs : dict
            Dictionary of kwargs to save.
        """
        d = kwargs.copy()
        d["eps"] = self.eps
        d["torch_dtype"] = self.torch_dtype
        d["importance_sampler"] = self.importance_nested_sampler
        save_to_json(d, os.path.join(self.output, "config.json"))

    def save_results(
        self,
        filename: str,
        extension: Optional[str] = None,
    ) -> None:
        """
        Save the results from sampling to a specific results file.

        Parameters
        ----------
        filename : str
            Name of file to save results to.
        extension : Optional[str]
            File extension to used. If not specified, it will be inferred from
            the filename.
        """
        d = self.ns.get_result_dictionary()

        d["posterior_samples"] = self.posterior_samples
        if hasattr(self, "initial_posterior_samples"):
            d["initial_posterior_samples"] = self.initial_posterior_samples

        ext = os.path.splitext(filename)[1].lstrip(".")
        if extension is None:
            if ext == "":
                raise RuntimeError(
                    "Must specify file extension if not present in filename!"
                )
            else:
                extension = ext
        elif ext == "":
            filename = ".".join([filename, extension])

        if extension == "json":
            d["posterior_samples"] = live_points_to_dict(
                d["posterior_samples"]
            )
            save_to_json(d, filename)
        elif extension in ["hdf5", "h5"]:
            save_dict_to_hdf5(d, filename)
        else:
            raise RuntimeError(f"Unknown file extension: {extension}")

    def terminate_run(self, code=None):
        """Terminate a sampling run.

        Parameters
        ----------
        code : int, optional
            Code passed to :code:`close_pool`
        """
        logger.warning("Terminating run")
        self.ns.close_pool(code=code)
        self.ns.checkpoint()

    def safe_exit(self, signum=None, frame=None):
        """
        Safely exit. This includes closing the multiprocessing pool.
        """
        logger.warning(f"Trying to safely exit with code {signum}")
        self.terminate_run(code=signum)
        logger.warning(f"Exiting with code: {self.exit_code}")
        sys.exit(self.exit_code)
