# -*- coding: utf-8 -*-
"""
Dynamic version of nessai
"""
import bisect
import logging

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from .nestedsampler import NestedSampler
from .evidence import _NSIntegralState


logger = logging.getLogger(__name__)


class DynamicNestedSampler(NestedSampler):

    importance_history = {}
    adjust_count = 0

    def draw_batch_from_point(self, point, n, logl_min=-np.inf,
                              logl_max=np.inf):
        """Draw a batch of samples using a point"""
        self.proposal.check_acceptance = True
        self.proposal.drawsize = int(n * 1.1)
        print(self.proposal.drawsize)
        n_gen = 0
        new_samples = np.zeros(n, dtype=self.proposal.x_dtype)

        while n_gen < n:
            self.proposal.populate(point, N=n)
            samples = self.proposal.samples
            in_bounds = (
                (samples['logL'] >= logl_min)
                & (samples['logL'] <= logl_max)
            )
            samples = samples[in_bounds]
            m = min(samples.size, n - n_gen)
            new_samples[n_gen:(n_gen + m)] = samples[:m]
            n_gen += m
            logger.info(
                f'Accepted {n_gen}/{n} samples in '
                f'[{logl_min:.3f}, {logl_max:.3f}]'
            )

        return new_samples

    def load_previous_proposal(self, iteration):
        """Load the proposal that was used for a given iteration.

        Parameters
        ----------
        iteration : int
            Iteration of the nested sampling algorithm at which the flow was
            used.
        """
        count = bisect.bisect_left(self.training_iterations, iteration) - 1
        logger.debug(f'Trying to load flow {count}')
        self.proposal.load_proposal_from_count(count)

    def determine_update_bounds(self, weights, fraction=0.9, pad=0.0):
        """Determine the bounds within which to update the samples.

        The bounds are determined using the importance weights for each
        sample and finding the samples with the lowest and highest logL that
        have an importance within a fraction of the maximum importance.

        Parameters
        ----------
        fraction : float, optional
            Fraction of the maximum importance.

        Returns
        -------
        int :
            Indices of the first and last nested samples that meet the
            criteria.
        """
        n = len(self.nested_samples)
        pad = int(pad * n)
        bounds = np.arange(n)[weights >= (fraction * max(weights))]
        bounds = (min(bounds) - pad, min(max(bounds) + pad, n - 1))
        logger.debug(f'Bounds: {bounds}')
        return bounds

    def add_samples(self, n=None, fraction=0.9, G=0.9):
        """Add sampler to the current run"""

        logger.info(f'Current number of samples: {len(self.nested_samples)}')
        logger.info(f'Current ln Z: {self.state.logZ}')
        logger.info(f'Current ESS: {self.state.effective_sample_size}')

        # TODO: compute n
        if n is None:
            n = self.nlive

        weights, _, _ = self.state.importance_weights(G=G)

        if not self.adjust_count:
            self.importance_history[self.adjust_count] = {
                'log_vols': self.state.log_vols[1:],
                'weights': weights
            }

        start, end = self.determine_update_bounds(weights, fraction=fraction)

        if end >= (len(self.nested_samples) - 1):
            logl_max = np.inf
            log_vol_max = -np.inf
        else:
            logl_max = self.nested_samples[end]['logL']
            log_vol_max = self.state.log_vols[start + 1]

        logl_min = self.nested_samples[start]['logL']
        log_vol_min = self.state.log_vols[start + 1]

        logger.debug(f'Log-likelihood range: [{logl_min}, {logl_max}]')

        worst_point = self.nested_samples[start]
        self.load_previous_proposal(start)

        samples = self.draw_batch_from_point(
            worst_point, n, logl_min=logl_min, logl_max=logl_max)
        samples = np.sort(samples, order='logL')

        new_state = _NSIntegralState(self.nlive)
        old_norm = self.state.log_w_norm
        # Update state to current point
        for i, s in enumerate(self.nested_samples[:start]):
            new_state.increment(s, log_w_norm=old_norm[i])

        new_nested_samples = self.nested_samples[:start]

        # Start adding together the sets of samples
        old, new = start, 0
        new_norm = logsumexp(samples['logW'])
        for i in range(end - start + n):
            log_w_norm = np.logaddexp(old_norm[old], new_norm)
            old_sample = self.nested_samples[old]
            new_sample = samples[new]

            if old_sample['logL'] == new_sample['logL']:
                new += 1
                continue
            elif old_sample['logL'] < new_sample['logL']:
                new_state.increment(old_sample, log_w_norm=log_w_norm)
                new_nested_samples.append(old_sample)
                old += 1
                if old >= len(self.nested_samples):
                    logger.warning('Ran out of old samples')
                    break
            else:
                new_state.increment(new_sample, log_w_norm=log_w_norm)
                new_nested_samples.append(new_sample)
                new += 1
                if new >= samples.size:
                    logger.debug('Ran out of new samples')
                    break
                new_norm = logsumexp(samples[new:]['logW'])

        if old < len(self.nested_samples):
            for i, s in enumerate(self.nested_samples[old:], start=old):
                new_state.increment(s, log_w_norm=old_norm[i])
            new_nested_samples += self.nested_samples[old:]
        elif new < samples.size:
            for i, s in enumerate(samples[new:], start=new):
                log_w_norm = logsumexp(samples[new:]['logW'])
                new_state.increment(s, log_w_norm=log_w_norm)
                new_nested_samples.append(s)

        new_state.finalise()
        logger.info(f'New nested samples: {len(new_nested_samples)}')
        logger.info(f'New evidence: {new_state.logZ}')
        logger.info(f'New ESS: {new_state.effective_sample_size}')

        self.state = new_state
        self.nested_samples = new_nested_samples

        self.adjust_count += 1
        self.importance_history[self.adjust_count] = {
            'log_vols': self.state.log_vols[1:],
            'weights': self.state.importance_weights(G=G)[0],
            'volume_range': (log_vol_min, log_vol_max),
            'likelihood_range': (logl_min, logl_max),
        }

        return new_state

    def plot_importance(self, filename=None):
        """Produce an importance plot"""
        fig = plt.figure()

        for k, v, in self.importance_history.items():
            lines = plt.plot(v['log_vols'], v['weights'], label=k)
            if v.get('volume_range', False):
                lvmin, lvmax = v['volume_range']
                plt.axvline(lvmin, c=lines[0].get_color())
                if np.isfinite(lvmax):
                    plt.axvline(lvmax, c=lines[0].get_color())
        plt.xlabel('log X')
        plt.ylabel('Importance')
        plt.legend()
        plt.tight_layout()

        if filename is None:
            return fig
        else:
            fig.savefig(filename)
            plt.close(fig)
