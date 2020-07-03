import os
import logging
import numpy as np

from .nestedsampler import NestedSampler
from .posterior import draw_posterior_samples


logger = logging.getLogger(__name__)


class FlowSampler:
    """
    Main class to handle running the nested sampler
    """

    def __init__(self, model, output='./', resume=True,
            resume_file='nested_sampler_resume.pkl', weights_file=None, **kwargs):

        self.output = output
        if resume and os.path.exists(output + resume_file):
            try:
                self.ns = NestedSampler.resume(output +  resume_file, model,
                        kwargs['flow_config'], weights_file)
            except EOFError:
                logger.error('Could not resume from pickle file')
                logger.error('Restarting run')
                self.ns = NestedSampler(model, output=output, resume_file=resume_file,
                        **kwargs)
            except Exception as e:
                raise RuntimeError(f'Could not resume sampler with error: {e}')
        else:
            self.ns = NestedSampler(model, output=output, resume_file=resume_file,
                    **kwargs)


    def run(self, resume=False, plot=True, save=True):
        """
        Run the nested samper
        """
        self.ns.initialise()
        self.logZ, self.nested_samples = self.ns.nested_sampling_loop(save=save)
        logger.info('Computing posterior samples')
        self.posterior_samples = draw_posterior_samples(self.nested_samples,
                self.ns.nlive)
        logger.info(f'Returned {self.posterior_samples.size} posterior samples')

        if save:
            np.savetxt(os.path.join(
                self.output,'posterior.dat'),
                self.posterior_samples,
                header=' '.join(self.posterior_samples.dtype.names),
                newline='\n',delimiter=' ')

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

