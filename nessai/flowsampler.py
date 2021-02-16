import json
import logging
import os
import signal
import sys

import numpy as np

from . import __version__ as version
from .livepoint import live_points_to_dict
from .nestedsampler import NestedSampler
from .posterior import draw_posterior_samples
from .utils import FPJSONEncoder, configure_threads


logger = logging.getLogger(__name__)


class FlowSampler:
    """
    Main class to handle running the nested sampler

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
        User-defined model
    output : str, optional (./)
        Output directory
    resume : bool, optional (True)
        If True try to resume the sampler is the resume file existis
    resume_file : str, optional
        File to resume sampler from
    weights_files : str, optional
        Weights files used to resume sampler that replaces the weights file
        saved internally.
    exit_code : int, optional (130)
        Exit code to use when forceably exiting the sampler.
    kwargs :
        Keyword arguments parsed to NestedSampler. See NestedSampler for
        details
    """
    def __init__(self, model, output='./', resume=True,
                 resume_file='nested_sampler_resume.pkl', weights_file=None,
                 exit_code=130, max_threads=1, **kwargs):

        configure_threads(
            max_threads=max_threads,
            pytorch_threads=kwargs.get('pytorch_threads', None),
            n_pool=kwargs.get('n_pool', None)
            )

        self.exit_code = exit_code

        self.output = output
        if resume:
            if not any((os.path.exists(self.output + f) for f in
                        [resume_file, resume_file + '.old'])):
                logger.warning('No files to resume from, starting sampling')
                self.ns = NestedSampler(model, output=output,
                                        resume_file=resume_file, **kwargs)
            else:
                try:
                    self.ns = NestedSampler.resume(output + resume_file,
                                                   model,
                                                   kwargs['flow_config'],
                                                   weights_file)
                except (FileNotFoundError, RuntimeError) as e:
                    logger.error(f'Could not load resume file from: {output} '
                                 f'with error {e}')
                    try:
                        resume_file += '.old'
                        self.ns = NestedSampler.resume(output + resume_file,
                                                       model,
                                                       kwargs['flow_config'],
                                                       weights_file)
                    except RuntimeError as e:
                        logger.error('Could not load old resume '
                                     f'file from: {output}')
                        raise RuntimeError('Could not resume sampler '
                                           f'with error: {e}')
        else:
            self.ns = NestedSampler(model, output=output,
                                    resume_file=resume_file, **kwargs)

        self.save_kwargs(kwargs)

        try:
            signal.signal(signal.SIGTERM, self.safe_exit)
            signal.signal(signal.SIGINT, self.safe_exit)
            signal.signal(signal.SIGALRM, self.safe_exit)
        except AttributeError:
            logger.critical('Can not set signal attributes on this system')

    def run(self, plot=True, save=True):
        """
        Run the nested samper

        Parameters
        ----------
        plot : bool, optional (True)
            Toggle plots produced once the sampler has converged
        save : bool, opitional (True)
            Toggle automatic saving of results
        """
        self.ns.initialise()
        self.logZ, self.nested_samples = \
            self.ns.nested_sampling_loop()
        logger.info((f'Total sampling time: {self.ns.sampling_time}'))

        logger.info('Starting post processing')
        logger.info('Computing posterior samples')
        self.posterior_samples = draw_posterior_samples(self.nested_samples,
                                                        self.ns.nlive)
        logger.info(f'Returned {self.posterior_samples.size} '
                    'posterior samples')

        if save:
            self.save_results(f'{self.output}/result.json')

        if plot:
            from nessai import plot

            plot.plot_likelihood_evaluations(
                    self.ns.likelihood_evaluations,
                    self.ns.nlive,
                    filename=f'{self.output}/likelihood_evaluations.png')

            plot.plot_live_points(self.posterior_samples,
                                  filename=(f'{self.output}/'
                                            'posterior_distribution.png'))

            plot.plot_indices(self.ns.insertion_indices, self.ns.nlive,
                              filename=f'{self.output}/insertion_indices.png')

            self.ns.state.plot(f'{self.output}/logXlogL.png')

    def save_kwargs(self, kwargs):
        """
        Save the dictionary of keyword arguments used.

        Parameters
        ----------
        kwargs : dict
            Dictionary of kwargs to save
        """
        d = kwargs.copy()
        with open(f'{self.output}/config.json', 'w') as wf:
            try:
                json.dump(d, wf, indent=4, cls=FPJSONEncoder)
            except TypeError:
                if 'flow_class' in d:
                    d['flow_class'] = str(d['flow_class'])
                    json.dump(d, wf, indent=4, cls=FPJSONEncoder)
            except Exception as e:
                raise e

    def save_results(self, filename):
        """
        Save the results from sampling to a specific file.

        Parameters
        ----------
        filename : str
            Name of file to save results to.
        """
        iterations = np.arange(len(self.ns.min_likelihood)) \
            * (self.ns.nlive // 10)
        iterations[-1] = self.ns.iteration
        d = dict()
        d['version'] = version
        d['history'] = dict(
                iterations=iterations,
                min_likelihood=self.ns.min_likelihood,
                max_likelihood=self.ns.max_likelihood,
                likelihood_evaluations=self.ns.likelihood_evaluations,
                logZ=self.ns.logZ_history,
                dZ=self.ns.dZ_history,
                mean_acceptance=self.ns.mean_acceptance_history,
                rolling_p=self.ns.rolling_p,
                population=dict(
                    iterations=self.ns.population_iterations,
                    acceptance=self.ns.population_acceptance
                    ),
                training_iterations=self.ns.training_iterations

                )
        d['insertion_indices'] = self.ns.insertion_indices
        d['nested_samples'] = live_points_to_dict(self.nested_samples)
        d['posterior_samples'] = live_points_to_dict(self.posterior_samples)
        d['log_evidence'] = self.ns.log_evidence
        d['information'] = self.ns.information
        d['sampling_time'] = self.ns.sampling_time.total_seconds()
        d['training_time'] = self.ns.training_time.total_seconds()
        d['population_time'] = self.ns.proposal_population_time.total_seconds()
        if (t := self.ns.likelihood_evaluation_time.total_seconds()):
            d['likelihood_evaluation_time'] = t

        with open(filename, 'w') as wf:
            json.dump(d, wf, indent=4, cls=FPJSONEncoder)

    def safe_exit(self, signum=None, frame=None):
        """
        Safely exit. This includes closing the multiprocessing pool.
        """
        logger.warning(f'Trying to safely exit with code {signum}')
        self.ns.proposal.close_pool(code=signum)
        self.ns.checkpoint()

        logger.warning(f'Exiting with code: {self.exit_code}')
        sys.exit(self.exit_code)
