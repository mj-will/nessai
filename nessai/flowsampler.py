# -*- coding: utf-8 -*-
"""
Main code that handles running and checkpoiting the sampler.
"""
import json
import logging
import os
import signal
import sys

from .livepoint import live_points_to_dict
from .samplers.nestedsampler import NestedSampler
from .posterior import draw_posterior_samples
from .utils import NessaiJSONEncoder, configure_threads


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
    weights_files : str, optional
        Weights files used to resume sampler that replaces the weights file
        saved internally.
    pytorch_threads : int
        Maximum number of threads to use for torch. If ``None`` torch uses all
        available threads.
    max_threads : int
        Deprecated and will be removed in a future release.
    signal_handling : bool
        Enable or disable signal handling.
    exit_code : int, optional
        Exit code to use when forceably exiting the sampler.
    close_pool : bool
        If True, the multiprocessing pool will be closed once the run method
        has been called. Disables the option in :code:`NestedSampler` if
        enabled.
    kwargs :
        Keyword arguments passed to
        :obj:`~nessai.samplers.nestedsampler.NestedSampler`.
    """

    def __init__(
        self,
        model,
        output=os.getcwd(),
        resume=True,
        resume_file="nested_sampler_resume.pkl",
        weights_file=None,
        signal_handling=True,
        exit_code=130,
        pytorch_threads=1,
        max_threads=None,
        close_pool=True,
        **kwargs,
    ):

        configure_threads(
            max_threads=max_threads,
            pytorch_threads=pytorch_threads,
        )

        self.exit_code = exit_code
        self.close_pool = close_pool

        self.output = os.path.join(output, "")
        if resume:
            if not resume_file:
                raise RuntimeError(
                    "`resume_file` must be specified if resume=True. "
                    f"Current value: {resume_file}"
                )
            if not any(
                (
                    os.path.exists(os.path.join(self.output, f))
                    for f in [resume_file, resume_file + ".old"]
                )
            ):
                logger.warning("No files to resume from, starting sampling")
                self.ns = NestedSampler(
                    model,
                    output=self.output,
                    resume_file=resume_file,
                    close_pool=not self.close_pool,
                    **kwargs,
                )
            else:
                try:
                    self.ns = NestedSampler.resume(
                        os.path.join(self.output, resume_file),
                        model,
                        flow_config=kwargs.get("flow_config"),
                        weights_file=weights_file,
                    )
                except (FileNotFoundError, RuntimeError) as e:
                    logger.error(
                        f"Could not load resume file from: {self.output} "
                        f"with error {e}"
                    )
                    try:
                        resume_file += ".old"
                        self.ns = NestedSampler.resume(
                            os.path.join(self.output, resume_file),
                            model,
                            flow_config=kwargs.get("flow_config"),
                            weights_file=weights_file,
                        )
                    except RuntimeError as e:
                        logger.error(
                            "Could not load old resume "
                            f"file from: {self.output}"
                        )
                        raise RuntimeError(
                            "Could not resume sampler " f"with error: {e}"
                        )
        else:
            self.ns = NestedSampler(
                model,
                output=self.output,
                resume_file=resume_file,
                close_pool=not self.close_pool,
                **kwargs,
            )

        self.save_kwargs(kwargs)

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
                "checkpoint when exitted."
            )

    @property
    def log_evidence(self):
        """Return the most recent log evidence"""
        return self.logZ

    @property
    def log_evidence_error(self):
        """Return the most recent log evidence error"""
        return self.logZ_error

    def run(
        self,
        plot=True,
        plot_indices=True,
        plot_posterior=True,
        plot_logXlogL=True,
        save=True,
        posterior_sampling_method=None,
        close_pool=None,
    ):
        """Run the nested sampler.

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
        if close_pool is None:
            close_pool = self.close_pool
        if posterior_sampling_method is None:
            posterior_sampling_method = "rejection_sampling"

        self.ns.initialise()
        self.logZ, self.nested_samples = self.ns.nested_sampling_loop()
        self.logZ_error = self.ns.state.log_evidence_error
        logger.info((f"Total sampling time: {self.ns.sampling_time}"))

        logger.info("Starting post processing")
        logger.info("Computing posterior samples")
        self.posterior_samples = draw_posterior_samples(
            self.nested_samples,
            log_w=self.ns.state.log_posterior_weights,
            method=posterior_sampling_method,
        )
        logger.info(
            f"Returned {self.posterior_samples.size} " "posterior samples"
        )

        if save:
            self.save_results(os.path.join(self.output, "result.json"))

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

    def save_kwargs(self, kwargs):
        """
        Save the dictionary of keyword arguments used.

        Uses an encoder class to handle numpy arrays.

        Parameters
        ----------
        kwargs : dict
            Dictionary of kwargs to save.
        """
        d = kwargs.copy()
        with open(os.path.join(self.output, "config.json"), "w") as wf:
            json.dump(d, wf, indent=4, cls=NessaiJSONEncoder)

    def save_results(self, filename):
        """
        Save the results from sampling to a specific JSON file.

        Parameters
        ----------
        filename : str
            Name of file to save results to.
        """
        d = self.ns.get_result_dictionary()
        d["posterior_samples"] = live_points_to_dict(self.posterior_samples)
        with open(filename, "w") as wf:
            json.dump(d, wf, indent=4, cls=NessaiJSONEncoder)

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
