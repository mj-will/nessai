# -*- coding: utf-8 -*-
"""
Main code that handles running and checkpoiting the sampler.
"""
import json
import logging
import os
import signal
import sys

from .nestedsampler import NestedSampler
from .importancesampler import ImportanceNestedSampler
from .livepoint import live_points_to_dict
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
    max_threads : int, optional
        Maximum number of threads to use. If ``None`` torch uses all available
        threads.
    exit_code : int, optional
        Exit code to use when forceably exiting the sampler.
    importance_sampler : bool
        If True the importance based nested sampler is used. This is disabled
        by default.
    kwargs :
        Keyword arguments passed to :obj:`~nessai.nestedsampler.NestedSampler`.
    """
    def __init__(
        self,
        model,
        output='./',
        resume=True,
        resume_file='nested_sampler_resume.pkl',
        weights_path=None,
        exit_code=130,
        max_threads=1,
        importance_sampler=False,
        **kwargs
    ):

        configure_threads(
            max_threads=max_threads,
            pytorch_threads=kwargs.get('pytorch_threads', None),
            n_pool=kwargs.get('n_pool', None)
            )

        if importance_sampler:
            SamplerClass = ImportanceNestedSampler
        else:
            SamplerClass = NestedSampler
        self.importance_sampler = importance_sampler

        self.exit_code = exit_code

        self.output = os.path.join(output, '')
        if resume:
            if not resume_file:
                raise RuntimeError(
                    '`resume_file` must be specified if resume=True. '
                    f'Current value: {resume_file}'
                )
            if not any((os.path.exists(os.path.join(self.output, f)) for f in
                        [resume_file, resume_file + '.old'])):
                logger.warning('No files to resume from, starting sampling')
                self.ns = SamplerClass(model, output=self.output,
                                       resume_file=resume_file, **kwargs)
            else:
                try:
                    self.ns = SamplerClass.resume(
                        os.path.join(self.output, resume_file),
                        model,
                        kwargs['flow_config'],
                        weights_path,
                    )
                except (FileNotFoundError, RuntimeError) as e:
                    logger.error(
                        f'Could not load resume file from: {self.output} '
                        f'with error {e}'
                    )
                    try:
                        resume_file += '.old'
                        self.ns = SamplerClass.resume(
                            os.path.join(self.output, resume_file),
                            model,
                            kwargs['flow_config'],
                            weights_path
                        )
                    except RuntimeError as e:
                        logger.error(
                            'Could not load old resume file from: '
                            f'{self.output}'
                        )
                        raise RuntimeError(
                            f'Could not resume sampler with error: {e}'
                        )
        else:
            self.ns = SamplerClass(model, output=self.output,
                                   resume_file=resume_file, **kwargs)

        self.save_kwargs(kwargs)

        try:
            signal.signal(signal.SIGTERM, self.safe_exit)
            signal.signal(signal.SIGINT, self.safe_exit)
            signal.signal(signal.SIGALRM, self.safe_exit)
        except AttributeError:
            logger.critical('Can not set signal attributes on this system')

    def run(
        self,
        plot=True,
        save=True,
        posterior_sampling_method=None,
        **kwargs,
    ):
        """Run the nested sampler.

        Will pick the correct run method given the configuration used.

        Parameters
        ----------
        plot : bool, optional
            Toggle plots produced once the sampler has converged
        save : bool, optional
            Toggle automatic saving of results
        """
        if self.importance_sampler:
            self.run_importance_sampler(
                plot=plot,
                save=save,
                posterior_sampling_method=posterior_sampling_method,
                **kwargs
            )
        else:
            self.run_standard_sampler(
                plot=plot,
                save=save,
                posterior_sampling_method=posterior_sampling_method,
                **kwargs
            )

    def run_standard_sampler(
        self,
        plot=True,
        save=True,
        posterior_sampling_method=None,
    ):
        """Run the standard nested sampler.

        Parameters
        ----------
        plot
            Enable or disable plotting. Independent of the value passed
            to the :code:`NestedSampler` object.
        save
            Enable or disable saving of a results file.
        posterior_sampling_method
            Method used for drawing posterior samples. Defaults to rejection
            sampling.

        """
        if self.importance_sampler:
            raise RuntimeError(
                'Cannot run standard sampler when importance_sampler=True'
            )

        if posterior_sampling_method is None:
            posterior_sampling_method = 'rejection_sampling'
        self.ns.initialise()
        self.logZ, self.nested_samples = \
            self.ns.nested_sampling_loop()
        logger.info((f'Total sampling time: {self.ns.sampling_time}'))

        logger.info('Starting post processing')
        logger.info('Computing posterior samples')
        self.posterior_samples = draw_posterior_samples(
            self.nested_samples,
            nlive=self.ns.nlive,
            method=posterior_sampling_method,
        )
        logger.info(f'Returned {self.posterior_samples.size} '
                    'posterior samples')

        if save:
            self.save_results(f'{self.output}/result.json')

        if plot:
            from nessai import plot

            plot.plot_live_points(self.posterior_samples,
                                  filename=(f'{self.output}/'
                                            'posterior_distribution.png'))

            plot.plot_indices(self.ns.insertion_indices, self.ns.nlive,
                              filename=f'{self.output}/insertion_indices.png')

            self.ns.state.plot(f'{self.output}/logXlogL.png')

    def run_importance_sampler(
        self,
        plot=True,
        save=True,
        posterior_sampling_method=None,
        redraw_samples=True,
        n_posterior_samples=None,
        compute_initial_posterior=True,
        **kwargs
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
        kwargs
            Keyword arguments passed to \
                :py:meth:`~nessai.importancesampler.ImportanceNestedSampler.draw_final_samples`
        """

        if not self.importance_sampler:
            raise RuntimeError(
                'Cannot run importance sampler when importance_sampler=False'
            )

        if posterior_sampling_method is None:
            posterior_sampling_method = 'importance_sampling'

        self.logZ, self.nested_samples = \
            self.ns.nested_sampling_loop()
        self.logZ_error = self.ns.state.log_evidence_error
        logger.info((f'Total sampling time: {self.ns.sampling_time}'))

        logger.info('Starting post processing')

        if redraw_samples:
            logger.info('Redrawing samples')
            self.initial_logZ = self.logZ
            self.initial_logZ_error = self.logZ_error
            self.logZ, self.final_samples = \
                self.ns.draw_final_samples(
                    n=n_posterior_samples, **kwargs,
                )
            self.logZ_error = self.ns.final_log_evidence_error

        logger.info('Computing posterior samples')

        if compute_initial_posterior or not redraw_samples:
            logger.debug('Computing initial posterior samples')
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
            f'Returned {self.posterior_samples.size} posterior samples'
        )

        if save:
            self.save_results(f'{self.output}/result.json')

        if plot:
            from nessai import plot
            plot.plot_live_points(
                self.posterior_samples,
                filename=os.path.join(
                    self.output, 'posterior_distribution.png'
                )
            )
            if redraw_samples and compute_initial_posterior:
                plot.plot_live_points(
                    self.posterior_samples,
                    filename=os.path.join(
                        self.output, 'initial_posterior_distribution.png'
                    )
                )

    @property
    def log_evidence(self):
        """Return the most recent log evidence"""
        return self.logZ

    @property
    def log_evidence_error(self):
        """Return the most recent log evidence error"""
        return self.logZ_error

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
        with open(f'{self.output}/config.json', 'w') as wf:
            try:
                json.dump(d, wf, indent=4, cls=NessaiJSONEncoder)
            except TypeError:
                if 'flow_class' in d:
                    d['flow_class'] = str(d['flow_class'])
                    json.dump(d, wf, indent=4, cls=NessaiJSONEncoder)
            except Exception as e:
                raise e

    def save_results(self, filename):
        """
        Save the results from sampling to a specific JSON file.

        Parameters
        ----------
        filename : str
            Name of file to save results to.
        """
        d = self.ns.get_result_dictionary()

        d['posterior_samples'] = live_points_to_dict(self.posterior_samples)
        if hasattr(self, 'initial_posterior_samples'):
            d['initial_posterior_samples'] = self.initial_posterior_samples

        with open(filename, 'w') as wf:
            json.dump(d, wf, indent=4, cls=NessaiJSONEncoder)

    def safe_exit(self, signum=None, frame=None):
        """
        Safely exit. This includes closing the multiprocessing pool.
        """
        logger.warning(f'Trying to safely exit with code {signum}')
        self.ns.close_pool(code=signum)
        self.ns.checkpoint()

        logger.warning(f'Exiting with code: {self.exit_code}')
        sys.exit(self.exit_code)
