import datetime
import json
import logging
import os
import time

import numpy as np

from .nestedsampler import NestedSampler
from .posterior import draw_posterior_samples
from .utils import NumpyEncoder, save_live_points


logger = logging.getLogger(__name__)


class FlowSampler:
    """
    Main class to handle running the nested sampler
    """

    def __init__(self, model, output='./', resume=True,
            resume_file='nested_sampler_resume.pkl', weights_file=None, **kwargs):

        self.output = output
        if resume:
            if not any((os.path.exists(self.output + f) for f in [resume_file,
                resume_file + '.old'])):
                logger.warning('No files to resume from, starting sampling')
                self.ns = NestedSampler(model, output=output,
                        resume_file=resume_file, **kwargs)
            else:
                try:
                    self.ns = NestedSampler.resume(output +  resume_file, model,
                            kwargs['flow_config'], weights_file)
                except (FileNotFoundError, RuntimeError) as e:
                    logger.error(f'Could not load resume file from: {output}')
                    try:
                        resume_file += '.old'
                        self.ns = NestedSampler.resume(output +  resume_file, model,
                                kwargs['flow_config'], weights_file)
                    except RuntimeError as e:
                        logger.error('Could not load old resume file from: {output}')
                        raise RuntimeError(f'Could not resume sampler with error: {e}')
        else:
            self.ns = NestedSampler(model, output=output, resume_file=resume_file,
                    **kwargs)

        self.save_kwargs(kwargs)

    def run(self, resume=False, plot=True, save=True):
        """
        Run the nested samper
        """
        self.ns.initialise()
        st = time.time()
        self.logZ, self.nested_samples = self.ns.nested_sampling_loop(save=save)
        logger.info(('Total sampling time: '
            f'{datetime.timedelta(seconds=time.time() - st)}'))
        logger.info('Computing posterior samples')
        self.posterior_samples = draw_posterior_samples(self.nested_samples,
                self.ns.nlive)
        logger.info(f'Returned {self.posterior_samples.size} posterior samples')

        if save:
            self.save_results(f'{self.output}/result.json')

        if plot:
            from flowproposal import plot

            plot.plot_likelihood_evaluations(self.ns.likelihood_evaluations,
                    self.ns.nlive,
                    filename=f'{self.output}/likelihood_evaluations.png')

            plot.plot_live_points(self.posterior_samples,
                    filename=f'{self.output}/posterior_distribution.png')

            plot.plot_indices(self.ns.insertion_indices, self.ns.nlive,
                    filename=f'{self.output}/insertion_indices.png')

            self.ns.state.plot(f'{self.output}/logXlogL.png')

    def save_kwargs(self, kwargs):
        """
        Save the key-word arguments used
        """
        d = kwargs.copy()
        with open(f'{self.output}/config.json', 'w') as wf:
            try:
                json.dump(d, wf, indent=4, cls=NumpyEncoder)
            except TypeError:
                d['flow_class'] = str(d['flow_class'])
                json.dump(d, wf, indent=4, cls=NumpyEncoder)
            except Exception as e:
                raise e


    def save_results(self, filename):
        """
        Save the results from sampling
        """
        iterations = (np.arange(len(self.ns.min_likelihood))) * (self.ns.nlive // 10)
        iterations[-1] = self.ns.iteration
        d = dict()
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
        d['nested_samples'] = self.nested_samples
        d['posterior_samples'] = self.posterior_samples

        with open(filename, 'w') as wf:
            json.dump(d, wf, indent=4, cls=NumpyEncoder)



