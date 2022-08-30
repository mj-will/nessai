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
    signal_handling : bool
        Enable or disable signal handling.
    exit_code : int, optional
        Exit code to use when forceably exiting the sampler.
    kwargs :
        Keyword arguments passed to
        :obj:`~nessai.samplers.nestedsampler.NestedSampler`.
    """

    def __init__(
        self,
        model,
        output="./",
        resume=True,
        resume_file="nested_sampler_resume.pkl",
        weights_file=None,
        signal_handling=True,
        exit_code=130,
        pytorch_threads=1,
        **kwargs,
    ):

        configure_threads(
            max_threads=kwargs.get("max_threads", None),
            pytorch_threads=pytorch_threads,
        )

        self.exit_code = exit_code

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
                model, output=self.output, resume_file=resume_file, **kwargs
            )

        self.save_kwargs(kwargs)

        if signal_handling:
            try:
                signal.signal(signal.SIGTERM, self.safe_exit)
                signal.signal(signal.SIGINT, self.safe_exit)
                signal.signal(signal.SIGALRM, self.safe_exit)
            except AttributeError:
                logger.critical("Cannot set signal attributes on this system")
        else:
            logger.warning(
                "Signal handling is disabled. nessai will not automatically "
                "checkpoint when exitted."
            )

    def run(self, plot=True, save=True):
        """
        Run the nested samper

        Parameters
        ----------
        plot : bool, optional
            Toggle plots produced once the sampler has converged
        save : bool, optional
            Toggle automatic saving of results
        """
        self.ns.initialise()
        self.logZ, self.nested_samples = self.ns.nested_sampling_loop()
        logger.info((f"Total sampling time: {self.ns.sampling_time}"))

        logger.info("Starting post processing")
        logger.info("Computing posterior samples")
        self.posterior_samples = draw_posterior_samples(
            self.nested_samples, self.ns.nlive
        )
        logger.info(
            f"Returned {self.posterior_samples.size} " "posterior samples"
        )

        if save:
            self.save_results(os.path.join(self.output, "result.json"))

        if plot:
            from nessai import plot

            plot.plot_live_points(
                self.posterior_samples,
                filename=os.path.join(
                    self.output, "posterior_distribution.png"
                ),
            )

            plot.plot_indices(
                self.ns.insertion_indices,
                self.ns.nlive,
                filename=os.path.join(self.output, "insertion_indices.png"),
            )

            self.ns.state.plot(os.path.join(self.output, "logXlogL.png"))

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

    def safe_exit(self, signum=None, frame=None):
        """
        Safely exit. This includes closing the multiprocessing pool.
        """
        logger.warning(f"Trying to safely exit with code {signum}")
        self.ns.model.close_pool(code=signum)
        self.ns.checkpoint()

        logger.warning(f"Exiting with code: {self.exit_code}")
        sys.exit(self.exit_code)
