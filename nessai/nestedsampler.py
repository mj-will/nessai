# -*- coding: utf-8 -*-
"""
Functions and objects related to the main nested sampling algorithm.
"""
from collections import deque
import datetime
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from .livepoint import get_dtype, DEFAULT_FLOAT_DTYPE
from .plot import plot_indices, plot_trace
from .evidence import _NSIntegralState
from .proposal import FlowProposal
from .utils import (
    safe_file_dump,
    compute_indices_ks_test,
    )

sns.set()
sns.set_style('ticks')

logger = logging.getLogger(__name__)


class NestedSampler:
    """
    Nested Sampler class.
    Initialisation arguments:

    Parameters
    ----------
    model: :obj:`nessai.model.Model`
        User defined model
    nlive : int, optional
        Number of live points.
    output : str
        Path for output
    stopping : float, optional
        Stop when remaining samples wouldn't change logZ estimate by this much.
    max_iteration : int, optional
        Maximum number of iterations to run before force sampler to stop.
        If stopping criteria is met before max. is reached sampler will stop.
    checkpointing : bool, optional
        Boolean to toggle checkpointing, must be enabled to resume the sampler.
        If false the sampler is still saved at the end of sampling.
    resume_file : str, optional
        If specified sampler will be resumed from this file. Still requires
        correct model.
    seed : int, optional
        seed for the initialisation of the pseudorandom chain
    plot : bool (True)
        Boolean to toggle plotting
    proposal_plots : bool (True)
        Boolean to enable additional plots for the population stage of the
        sampler. Overwritten by plot.
    prior_sampling : bool (False)
        produce nlive samples from the prior.
    analytic_priors : bool (False)
        Boolean that indicates that the `new_point` method in the model
        draws directly from the priors meaning rejection sampling is not
        needed.
    maximum_uninformed : int (1000)
        Maximum number of iterations before forcing the sampler to switch to
        using the proposal method with the flow.
    uninformed_proposal : :obj:`nessai.proposal.Proposal`: (None)
        Class to use for initial sampling before training the flow. If
        None RejectionProposal or AnalyticProposal are used depending if
        `analytic_priors` is False or True.
    uninformed_acceptance_threshold : float (None)
        Acceptance threshold for initialising sampling, if acceptance falls
        below this value sampler switches to flow-based proposal. If None
        then value is set to 10 times `acceptance_threshold`
    uninformed_proposal_kwargs : dict, ({})
        Dictionary of keyword argument to pass to the class use for
        the initial sampling when it is initialised.
    flow_class : :obj:`nessai.proposal.FlowProposal`
        Class to use for flow-based proposal method
    flow_config : dict ({})
        Dictionary used to configure instance of `nessai.flowmodel.FlowModel`,
        this includes configuring the normalising flow and the training.
    training_frequency : int (None)
        Number of iterations between re-training the flow. If None flow
        is only re-trained based on other criteria.
    train_on_empty : bool (True)
        If true the flow is retrained every time the proposal pool is
        empty. If false it is only training according to the other criteria.
    cooldown : int (100)
        Minimum number of iterations between training. Can be overridden if
        `train_on_empty=True` and the pool is empty.
    memory : int, False (False)
        Number of old live points to use in training. If False only the current
        live points are used.
    reset_weights : bool, int, (False)
        Boolean to toggle resetting the flow weights whenever re-training.
        If an integer is specified the flow is reset every nth time it is
        trained.
    reset_permutations: bool, int, (False)
        Boolean to toggle resetting the permutation layers in the flow whenever
        re-training. If an integer is specified the flow is reset every nth
        time it is trained.
    reset_acceptance : bool, (True)
        If true use mean acceptance of samples produced with current flow
        as a criteria for retraining
    retrain_acceptance : bool (False)
        Force the flow to be reset if the acceptance falls below the acceptance
        threshold. Requires `reset_acceptance=True`
    acceptance_threshold : float (0.01)
        Threshold to determine if the flow should be retrained, will not
        retrain if cooldown is not satisfied.
    kwargs :
        Keyword arguments passed to the flow proposal class
    """

    def __init__(
        self,
        model,
        nlive=2000,
        output=None,
        stopping=0.1,
        max_iteration=None,
        checkpointing=True,
        checkpoint_on_training=False,
        resume_file=None,
        seed=None,
        n_pool=None,
        plot=True,
        proposal_plots=False,
        prior_sampling=False,
        analytic_priors=False,
        maximum_uninformed=None,
        uninformed_proposal=None,
        uninformed_acceptance_threshold=None,
        uninformed_proposal_kwargs=None,
        flow_class=None,
        flow_config=None,
        training_frequency=None,
        train_on_empty=True,
        cooldown=200,
        memory=False,
        reset_weights=False,
        reset_permutations=False,
        retrain_acceptance=True,
        reset_acceptance=False,
        acceptance_threshold=0.01,
        **kwargs
    ):

        logger.info('Initialising nested sampler')

        self.info_enabled = logger.isEnabledFor(logging.INFO)

        model.verify_model()

        self.model = model
        self.nlive = nlive
        self.n_pool = n_pool
        self.live_points = None
        self.prior_sampling = prior_sampling
        self.setup_random_seed(seed)
        self.accepted = 0
        self.rejected = 1
        self.initialised = False

        self.checkpointing = checkpointing
        self.checkpoint_on_training = checkpoint_on_training
        self.iteration = 0
        self.acceptance_history = deque(maxlen=(nlive // 10))
        self.mean_acceptance_history = []
        self.block_acceptance = 1.
        self.mean_block_acceptance = 1.
        self.block_iteration = 0
        self.retrain_acceptance = retrain_acceptance
        self.reset_acceptance = reset_acceptance

        self.insertion_indices = []
        self.rolling_p = []

        self.resumed = False
        self.tolerance = stopping
        self.condition = np.inf
        self.logLmin = -np.inf
        self.logLmax = -np.inf
        self.nested_samples = []
        self.logZ = None
        self.state = _NSIntegralState(self.nlive, track_gradients=plot)
        self.plot = plot
        self.resume_file = self.setup_output(output, resume_file)
        self.output = output

        # Timing
        self.training_time = datetime.timedelta()
        self.sampling_time = datetime.timedelta()
        self.sampling_start_time = datetime.datetime.now()

        # Resume flags
        self.completed_training = True
        self.finalised = False

        # History
        self.likelihood_evaluations = []
        self.training_iterations = []
        self.min_likelihood = []
        self.max_likelihood = []
        self.logZ_history = []
        self.dZ_history = []
        self.population_acceptance = []
        self.population_radii = []
        self.population_iterations = []
        self.checkpoint_iterations = []

        self.acceptance_threshold = acceptance_threshold

        self.train_on_empty = train_on_empty
        self.cooldown = cooldown
        self.memory = memory

        self.configure_max_iteration(max_iteration)
        self.configure_flow_reset(reset_weights, reset_permutations)
        self.configure_training_frequency(training_frequency)

        if uninformed_proposal_kwargs is None:
            uninformed_proposal_kwargs = {}
        self.configure_uninformed_proposal(uninformed_proposal,
                                           analytic_priors,
                                           maximum_uninformed,
                                           uninformed_acceptance_threshold,
                                           **uninformed_proposal_kwargs)
        self.configure_flow_proposal(flow_class, flow_config, proposal_plots,
                                     **kwargs)

        # Uninformed proposal is used for prior sampling
        # If maximum uninformed is greater than 0, the it will be used for
        # another n iterations or until it becomes inefficient

        self.store_live_points = False
        if self.store_live_points:
            self.live_points_dir = f'{self.output}/live_points/'
            os.makedirs(self.live_points_dir, exist_ok=True)
            self.replacement_points = []

    @property
    def log_evidence(self):
        return self.state.logZ

    @property
    def information(self):
        return self.state.info[-1]

    @property
    def likelihood_calls(self):
        return self.model.likelihood_evaluations

    @property
    def likelihood_evaluation_time(self):
        t = self._uninformed_proposal.logl_eval_time
        t += self._flow_proposal.logl_eval_time
        return t

    @property
    def proposal_population_time(self):
        t = self._uninformed_proposal.population_time
        t += self._flow_proposal.population_time
        return t

    @property
    def acceptance(self):
        return self.iteration / self.likelihood_calls

    @property
    def current_sampling_time(self):
        if self.finalised:
            return self.sampling_time
        else:
            return self.sampling_time \
                    + (datetime.datetime.now() - self.sampling_start_time)

    @property
    def last_updated(self):
        """Last time the normalising flow was retrained"""
        if self.training_iterations:
            return self.training_iterations[-1]
        else:
            return 0

    @property
    def mean_acceptance(self):
        """
        Mean acceptance of the last nlive // 10 points
        """
        if self.acceptance_history:
            return np.mean(self.acceptance_history)
        else:
            return np.nan

    def configure_max_iteration(self, max_iteration):
        """Configure the maximum iteration.

        If None then no maximum is set.

        Parameter
        ---------
        max_iteration : int, None
            Maximum iteration.
        """
        if max_iteration is None:
            self.max_iteration = np.inf
        else:
            self.max_iteration = max_iteration

    def configure_training_frequency(self, training_frequency):
        """Configure the training frequency.

        If None, 'inf' or 'None' flow will only train when empty.
        """
        if training_frequency in [None, 'inf', 'None']:
            logger.warning('Proposal will only train when empty')
            self.training_frequency = np.inf
        else:
            self.training_frequency = training_frequency

    def configure_uninformed_proposal(self,
                                      uninformed_proposal,
                                      analytic_priors,
                                      maximum_uninformed,
                                      uninformed_acceptance_threshold,
                                      **kwargs):
        """
        Setup the uninformed proposal method (is NOT trained)

        Parameters
        ----------
        uninformed_proposal : None or obj
            Class to use for uninformed proposal
        analytic_priors : bool
            If True `AnalyticProposal` is used to directly sample from the
            priors rather than using rejection sampling.
        maximum_uninformed : {False, None, int, float}
            Maximum number of iterations before switching to FlowProposal.
            If None, two times nlive is used. If False uninformed sampling is
            not used.
        uninformed_acceptance_threshold :  float or None:
            Threshold to use for uninformed proposal, once reached proposal
            method will switch. If None acceptance_threshold is used if
            greater than 0.1 else 10 x acceptance_threshold is used.
        kwargs
            Kwargs are passed to init method for uninformed proposal class
        """
        if maximum_uninformed is None:
            self.uninformed_sampling = True
            self.maximum_uninformed = 2 * self.nlive
        elif not maximum_uninformed:
            self.uninformed_sampling = False
            self.maximum_uninformed = 0
        else:
            self.uninformed_sampling = True
            self.maximum_uninformed = float(maximum_uninformed)

        if uninformed_acceptance_threshold is None:
            if self.acceptance_threshold < 0.1:
                self.uninformed_acceptance_threshold = \
                    10 * self.acceptance_threshold
            else:
                self.uninformed_acceptance_threshold = \
                    self.acceptance_threshold
        else:
            self.uninformed_acceptance_threshold = \
                uninformed_acceptance_threshold

        if uninformed_proposal is None:
            if analytic_priors:
                from .proposal import AnalyticProposal as uninformed_proposal
            else:
                from .proposal import RejectionProposal as uninformed_proposal
                kwargs['poolsize'] = self.nlive

        logger.debug(f'Using uninformed proposal: {uninformed_proposal}')
        logger.debug(f'Parsing kwargs to uninformed proposal: {kwargs}')
        self._uninformed_proposal = uninformed_proposal(
            self.model, n_pool=self.n_pool, **kwargs)

    def configure_flow_proposal(self, flow_class, flow_config, proposal_plots,
                                **kwargs):
        """
        Set up the flow-based proposal method

        Parameters
        ----------
        flow_class : None or obj or str
            Class to use for proposal. If None FlowProposal is used.
        flow_config : dict
            Configuration dictionary passed to the class.
        proposal_plots : bool or str
            Configuration of plotting in proposal class.
        **kwargs :
            Kwargs passed to init function.
        """
        proposal_output = self.output + '/proposal/'

        if not self.plot:
            proposal_plots = False

        if flow_class is not None:
            if isinstance(flow_class, str):
                flow_class = flow_class.lower()
                if flow_class == 'gwflowproposal':
                    from .gw.proposal import GWFlowProposal as flow_class
                elif flow_class == 'augmentedgwflowproposal':
                    from .gw.proposal import (
                        AugmentedGWFlowProposal as flow_class)
                elif flow_class == 'legacygwflowproposal':
                    from .gw.legacy import LegacyGWFlowProposal as flow_class
                elif flow_class == 'flowproposal':
                    flow_class = FlowProposal
                elif flow_class == 'augmentedflowproposal':
                    from .proposal import AugmentedFlowProposal
                    flow_class = AugmentedFlowProposal
                else:
                    raise ValueError(f'Unknown flow class: {flow_class}')
            elif not issubclass(flow_class, FlowProposal):
                raise RuntimeError('Flow class must be string or class that '
                                   'inherits from FlowProposal')
        else:
            flow_class = FlowProposal

        if kwargs.get('poolsize', None) is None:
            kwargs['poolsize'] = self.nlive

        logger.debug(f'Using flow class: {flow_class}')
        logger.info(f'Parsing kwargs to FlowProposal: {kwargs}')
        self._flow_proposal = flow_class(
            self.model, flow_config=flow_config, output=proposal_output,
            plot=proposal_plots, n_pool=self.n_pool, **kwargs)

    def setup_output(self, output, resume_file=None):
        """
        Set up the output folder

        Parameters
        ----------
        output : str
            Directory where the results will be stored
        resume_file : optional
            Specific file to use for checkpointing. If not specified the
            default is used (nested_sampler_resume.pkl)

        Returns
        -------
        resume_file : str
            File used for checkpointing
        """
        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)

        if resume_file is None:
            resume_file = os.path.join(output, "nested_sampler_resume.pkl")
        else:
            resume_file = os.path.join(output, resume_file)

        if self.plot:
            os.makedirs(output + '/diagnostics/', exist_ok=True)

        return resume_file

    def setup_random_seed(self, seed):
        """
        initialise the random seed
        """
        self.seed = seed
        if self.seed is not None:
            logger.debug(f'Setting random seed to {seed}')
            np.random.seed(seed=self.seed)
            torch.manual_seed(self.seed)

    def configure_flow_reset(self, reset_weights, reset_permutations):
        """Configure how often the flow parameters are reset.

        Values are converted to floats.

        Parameters
        ----------
        reset_weights : int, float or bool
            Frequency with which the weights will be reset.
        reset_permutations : int, float or bool
            Frequency with which the permutations will be reset.
        """
        if isinstance(reset_weights, (int, float)):
            self.reset_weights = float(reset_weights)
        else:
            raise TypeError(
                '`reset_weights` must be a bool, int or float')
        if isinstance(reset_permutations, (int, float)):
            self.reset_permutations = float(reset_permutations)
        else:
            raise TypeError(
                '`reset_permutations` must be a bool, int or float')

    def check_insertion_indices(self, rolling=True, filename=None):
        """
        Checking the distribution of the insertion indices either during
        the nested sampling run (rolling=True) or for the whole run
        (rolling=False).
        """
        if rolling:
            indices = self.insertion_indices[-self.nlive:]
        else:
            indices = self.insertion_indices

        D, p = compute_indices_ks_test(indices, self.nlive)

        if p is not None:
            if rolling:
                logger.warning(f'Rolling KS test: D={D:.4}, p-value={p:.4}')
                self.rolling_p.append(p)
            else:
                logger.warning(f'Final KS test: D={D:.4}, p-value={p:.4}')

        if filename is not None:
            np.savetxt(os.path.join(self.output, filename),
                       self.insertion_indices, newline='\n', delimiter=' ')

    def log_likelihood(self, x):
        """
        Wrapper for the model likelihood so evaluations are counted
        """
        return self.model.log_likelihood(x)

    def yield_sample(self, oldparam):
        """
        Draw points and applying rejection sampling
        """
        while True:
            counter = 0
            while True:
                counter += 1
                newparam = self.proposal.draw(oldparam.copy())

                # Prior is computed in the proposal
                if newparam['logP'] != -np.inf:
                    if not newparam['logL']:
                        newparam['logL'] = \
                            self.model.evaluate_log_likelihood(newparam)
                    if newparam['logL'] > self.logLmin:
                        self.logLmax = max(self.logLmax, newparam['logL'])
                        oldparam = newparam.copy()
                        break
                # Only here if proposed and then empty
                # This returns the old point and allows for a training check
                if not self.proposal.populated:
                    break
            yield counter, oldparam

    def insert_live_point(self, live_point):
        """
        Insert a live point
        """
        # This is the index including the current worst point, so final index
        # is one less, otherwise index=0 would never be possible
        index = np.searchsorted(self.live_points['logL'], live_point['logL'])
        self.live_points[:index - 1] = self.live_points[1:index]
        self.live_points[index - 1] = live_point
        return index - 1

    def consume_sample(self):
        """
        Replace a sample for single thread
        """
        worst = self.live_points[0].copy()
        self.logLmin = worst['logL']
        self.state.increment(worst['logL'])
        self.nested_samples.append(worst)

        self.condition = np.logaddexp(self.state.logZ,
                                      self.logLmax
                                      - self.iteration / float(self.nlive)) \
            - self.state.logZ

        # Replace the points we just consumed with the next acceptable ones
        # Make sure we are mixing the chains
        self.iteration += 1
        self.block_iteration += 1
        count = 0

        while(True):
            c, proposed = next(self.yield_sample(worst))
            count += c

            if proposed['logL'] > self.logLmin:
                # Assuming point was proposed
                # replace worst point with new one
                index = self.insert_live_point(proposed)
                self.insertion_indices.append(index)
                self.accepted += 1
                self.block_acceptance += 1 / count
                self.acceptance_history.append(1 / count)
                break
            else:
                # Only get here if the yield sample returns worse point
                # which can only happen if the pool is empty
                self.rejected += 1
                self.check_state()
                # if retrained whilst proposing a sample then update the
                # iteration count since will be zero otherwise
                if not self.block_iteration:
                    self.block_iteration += 1

        self.mean_block_acceptance = self.block_acceptance \
            / self.block_iteration

        if self.info_enabled:
            logger.info(f"{self.iteration:5d}: n: {count:3d} "
                        f"b_acc: {self.mean_block_acceptance:.3f} "
                        f"H: {self.state.info[-1]:.2f} "
                        f"logL: {self.logLmin:.5f} --> {proposed['logL']:.5f} "
                        f"dZ: {self.condition:.3f} "
                        f"logZ: {self.state.logZ:.3f} "
                        f"+/- {np.sqrt(self.state.info[-1] / self.nlive):.3f} "
                        f"logLmax: {self.logLmax:.2f}")

    def populate_live_points(self):
        """
        Initialise the pool of live points.
        """
        i = 0
        live_points = np.empty(self.nlive,
                               dtype=get_dtype(self.model.names,
                                               DEFAULT_FLOAT_DTYPE))

        with tqdm(total=self.nlive, desc='Drawing live points') as pbar:
            while i < self.nlive:
                while i < self.nlive:
                    count, live_point = next(
                            self.yield_sample(self.model.new_point()))
                    if np.isnan(live_point['logL']):
                        logger.warning(
                            'Likelihood function returned NaN for '
                            f'live_point {live_point}'
                        )
                        logger.warning(
                            'You may want to check your likelihood function'
                        )
                        break
                    if (
                        np.isfinite(live_point['logP'])
                        and np.isfinite(live_point['logL'])
                    ):
                        live_points[i] = live_point
                        i += 1
                        pbar.update()
                        break

        self.live_points = np.sort(live_points, order='logL')
        if self.store_live_points:
            np.savetxt(self.live_points_dir + '/initial_live_points.dat',
                       self.live_points,
                       header='\t'.join(self.live_points.dtype.names))

    def initialise(self, live_points=True):
        """
        Initialise the nested sampler

        Parameters
        ----------
        live_points : bool, optional (True)
            If true and there are no live points, new live points are
            drawn using `populate_live_points` else all other initialisation
            steps are complete but live points remain empty.
        """
        flags = [False] * 3
        if not self._flow_proposal.initialised:
            self._flow_proposal.initialise()
            flags[0] = True

        if not self._uninformed_proposal.initialised:
            self._uninformed_proposal.initialise()
            flags[1] = True

        if (
            self.iteration < self.maximum_uninformed
            and self.uninformed_sampling
        ):
            self.proposal = self._uninformed_proposal
        else:
            self.proposal = self._flow_proposal

        self.proposal.configure_pool()

        if live_points and self.live_points is None:
            self.populate_live_points()
            flags[2] = True

        if self.condition > self.tolerance:
            self.finalised = False

        if all(flags):
            self.initialised = True

    def check_proposal_switch(self, force=False):
        """
        Check if the proposal should be switch from uninformed to
        flowproposal given the current state.

        If the flow proposal is already in use, no changes are made.

        Parameters
        ----------
        force : bool, optional
            If True proposal is forced to switch.

        Returns
        -------
        bool
            Flag to indicated if proposal was switched
        """
        if (
            (self.mean_acceptance < self.uninformed_acceptance_threshold)
            or (self.iteration >= self.maximum_uninformed)
            or force
        ):
            if self.proposal is self._flow_proposal:
                logger.warning('Already using flowproposal')
                return True
            logger.warning('Switching to FlowProposal')
            # Make sure the pool is closed
            if self.proposal.pool is not None:
                self.proposal.close_pool()
            self.proposal = self._flow_proposal
            if self.proposal.n_pool is not None:
                self.proposal.configure_pool()
            self.proposal.ns_acceptance = self.mean_block_acceptance
            self.uninformed_sampling = False
            return True
        # If using uninformed sampling, don't check training
        else:
            return False

    def check_training(self):
        """
        Check if the normalising flow should be trained

        Checks that can force training:
            - Training was previously stopped before completion
            - The pool is empty and the proposal was not in the process
              of populating when stopped.
        Checks that cannot force training is still on cooldown:
            - Acceptance falls below threshold and `retrain_acceptance` is
              true
            - The number of iterations since last training is equal to the
              training frequency

        Returns
        -------
        train : bool
            Try to train if true
        force : bool
            Force the training irrespective of cooldown
        """
        if not self.completed_training:
            logger.debug('Training flow (resume)')
            return True, True
        elif (not self.proposal.populated and
                self.train_on_empty and
                not self.proposal.populating):
            logger.debug('Training flow (proposal empty)')
            return True, True
        elif (self.mean_block_acceptance < self.acceptance_threshold and
                self.retrain_acceptance):
            logger.debug('Training flow (acceptance)')
            return True, False
        elif (self.iteration - self.last_updated) == self.training_frequency:
            logger.debug('Training flow (iteration)')
            return True, False
        else:
            return False, False

    def check_flow_model_reset(self):
        """
        Check if the normalising flow model should be reset.

        Checks acceptance if `reset_acceptance` is True and always checks
        how many times the flow has been trained.

        Flow will not be reset if it has not been trained. To force a reset
        manually call `proposal.reset_model_weights`.
        """
        if not self.proposal.training_count:
            return

        if (self.reset_acceptance
                and self.mean_block_acceptance < self.acceptance_threshold):
            self.proposal.reset_model_weights(weights=True, permutations=True)
            return

        if (self.reset_weights and
                not (self.proposal.training_count % self.reset_weights)):
            self.proposal.reset_model_weights(weights=True)

        if (self.reset_permutations and
                not (self.proposal.training_count % self.reset_permutations)):
            self.proposal.reset_model_weights(weights=False, permutations=True)

    def train_proposal(self, force=False):
        """
        Try to train the proposal. Proposal will not train if cooldown is not
        exceeded unless force is True.

        Parameters
        ----------
        force : bool
            Override training checks
        """
        if (self.iteration - self.last_updated < self.cooldown and not force):
            logger.debug('Not training, still cooling down!')
        else:
            self.completed_training = False
            self.check_flow_model_reset()

            training_data = self.live_points.copy()
            if self.memory and (len(self.nested_samples) >= self.memory):
                training_data = np.concatenate([
                    training_data, self.nested_samples[-self.memory:].copy()])

            st = datetime.datetime.now()
            self.proposal.train(training_data)
            self.training_time += (datetime.datetime.now() - st)
            self.training_iterations.append(self.iteration)

            self.block_iteration = 0
            self.block_acceptance = 0.
            self.completed_training = True
            if self.checkpoint_on_training:
                self.checkpoint(periodic=True)

    def check_state(self, force=False):
        """
        Check if state should be updated prior to drawing a new sample

        Force will override the cooldown mechanism.
        """

        if self.uninformed_sampling:
            if self.check_proposal_switch():
                force = True
            else:
                return
        # General override
        train = False
        if force:
            train = True
            logger.debug('Training flow (force)')
        elif not train:
            train, force = self.check_training()

        if train or force:
            self.train_proposal(force=force)

    def plot_state(self, filename=None):
        """
        Produce plots with the current state of the nested sampling run.
        Plots are saved to the output directory specified at initialisation.

        Parameters
        ----------
        filename : str, optional
            If specified the figure will be saved, otherwise the figure is
            returned.
        """

        fig, ax = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
        ax = ax.ravel()
        it = (np.arange(len(self.min_likelihood))) * (self.nlive // 10)
        it[-1] = self.iteration

        colours = ['#4575b4', '#d73027', '#fad117']

        ls = ['-', '--', ':']

        for t in self.training_iterations:
            for a in ax:
                a.axvline(t, ls='-', color='lightgrey')

        if not self.train_on_empty:
            for p in self.population_iterations:
                for a in ax:
                    a.axvline(p, ls='-', color='tab:orange')

        for i in self.checkpoint_iterations:
            for a in ax:
                a.axvline(i, ls='-', color='tab:pink')

        ax[0].plot(it, self.min_likelihood, label='Min logL',
                   c=colours[0], ls=ls[0])
        ax[0].plot(it, self.max_likelihood, label='Max logL',
                   c=colours[1], ls=ls[1])
        ax[0].set_ylabel('logL')
        ax[0].legend(frameon=False)

        if self.state.track_gradients:
            g = np.min([len(self.state.gradients), self.iteration])
            ax[1].plot(np.arange(g), np.abs(self.state.gradients[:g]),
                       c=colours[0], label='Gradient')
        else:
            logger.warning('Gradients were not saved, skipping.')
        ax[1].set_ylabel(r'$|d\log L/d \log X|$')
        ax[1].set_yscale('log')

        ax[2].plot(it, self.likelihood_evaluations, c=colours[0], ls=ls[0],
                   label='Evaluations')
        ax[2].set_ylabel('logL evaluations')

        ax[3].plot(it, self.logZ_history, label='logZ', c=colours[0], ls=ls[0])
        ax[3].set_ylabel('logZ')
        ax[3].legend(frameon=False)

        ax_dz = plt.twinx(ax[3])
        ax_dz.plot(it, self.dZ_history, label='dZ', c=colours[1], ls=ls[1])
        ax_dz.set_ylabel('dZ')
        handles, labels = ax[3].get_legend_handles_labels()
        handles_dz, labels_dz = ax_dz.get_legend_handles_labels()
        ax[3].legend(handles + handles_dz, labels + labels_dz, frameon=False)

        ax[4].plot(it, self.mean_acceptance_history, c=colours[0],
                   label='Proposal')
        ax[4].plot(self.population_iterations, self.population_acceptance,
                   c=colours[1], ls=ls[1], label='Population')
        ax[4].set_ylabel('Acceptance')
        ax[4].set_ylim((-0.1, 1.1))
        handles, labels = ax[4].get_legend_handles_labels()

        ax_r = plt.twinx(ax[4])
        ax_r.plot(self.population_iterations, self.population_radii,
                  label='Radius', color=colours[2], ls=ls[2])
        ax_r.set_ylabel('Population radius')
        handles_r, labels_r = ax_r.get_legend_handles_labels()
        ax[4].legend(handles + handles_r, labels + labels_r, frameon=False)

        if len(self.rolling_p):
            it = (np.arange(len(self.rolling_p)) + 1) * self.nlive
            ax[5].plot(it, self.rolling_p, 'o', c=colours[0], label='p-value')
        ax[5].set_ylabel('p-value')
        ax[5].set_ylim([-0.1, 1.1])

        ax[-1].set_xlabel('Iteration')

        fig.suptitle(f'Sampling time: {self.current_sampling_time}',
                     fontsize=16)

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig

    def plot_trace(self, filename=None):
        """
        Make trace plots for the nested samples.

        Parameters
        ----------
        filename : str, optional
            If filename is None, the figure is returned. Else the figure
            is saved with that file name.
        """
        if self.nested_samples:
            fig = plot_trace(self.state.log_vols[1:], self.nested_samples,
                             filename=filename)
            return fig
        else:
            logger.warning('Could not produce trace plot. No nested samples!')

    def plot_insertion_indices(self, filename=None, **kwargs):
        """
        Make a plot of all the insertion indices.

        Parameters
        ----------
        filename : str, optional
            If filename is None, the figure is returned. Else the figure
            is saved with that file name.
        kwargs :
            Keyword arguments passed to `nessai.plot.plot_indices`.
        """
        return plot_indices(
            self.insertion_indices,
            self.nlive,
            filename=filename,
            **kwargs
        )

    def update_state(self, force=False):
        """
        Update state after replacing a live point
        """
        # Check if acceptance is not None, this indicates the proposal
        # was populated
        if not self.proposal._checked_population:
            self.population_acceptance.append(
                self.proposal.population_acceptance)
            self.population_radii.append(self.proposal.r)
            self.population_iterations.append(self.iteration)
            self.proposal._checked_population = True

        if not (self.iteration % (self.nlive // 10)) or force:
            self.likelihood_evaluations.append(
                    self.model.likelihood_evaluations)
            self.min_likelihood.append(self.logLmin)
            self.max_likelihood.append(self.logLmax)
            self.logZ_history.append(self.state.logZ)
            self.dZ_history.append(self.condition)
            self.mean_acceptance_history.append(self.mean_acceptance)

        if not (self.iteration % self.nlive) or force:
            logger.warning(
                f"it: {self.iteration:5d}: "
                f"n eval: {self.likelihood_calls} "
                f"H: {self.state.info[-1]:.2f} "
                f"dZ: {self.condition:.3f} logZ: {self.state.logZ:.3f} "
                f"+/- {np.sqrt(self.state.info[-1] / self.nlive):.3f} "
                f"logLmax: {self.logLmax:.2f}")
            if self.checkpointing:
                self.checkpoint(periodic=True)
            if not force:
                self.check_insertion_indices()
                if self.plot:
                    plot_indices(self.insertion_indices[-self.nlive:],
                                 self.nlive,
                                 plot_breakdown=False,
                                 filename=(f'{self.output}/diagnostics/'
                                           'insertion_indices_'
                                           f'{self.iteration}.png'))

            if self.plot:
                self.plot_state(filename=f'{self.output}/state.png')
                self.plot_trace(filename=f'{self.output}/trace.png')

            if self.uninformed_sampling:
                self.block_acceptance = 0.
                self.block_iteration = 0

        self.proposal.ns_acceptance = self.mean_block_acceptance

    def checkpoint(self, periodic=False):
        """
        Checkpoint the classes internal state

        Parameters
        ----------
        periodic : bool
            Indicates if the checkpoint is regular periodic checkpointing
            or forced by a signal. If forced by a signal, it will show up on
            the state plot.
        """
        if not periodic:
            self.checkpoint_iterations += [self.iteration]
        self.sampling_time += \
            (datetime.datetime.now() - self.sampling_start_time)
        logger.critical('Checkpointing nested sampling')
        safe_file_dump(self, self.resume_file, pickle, save_existing=True)
        self.sampling_start_time = datetime.datetime.now()

    def check_resume(self):
        """
        Check the normalising flow is correctly configured is the sampler
        was resumed.
        """
        if self.resumed:
            if self.uninformed_sampling is False:
                self.check_proposal_switch(force=True)
            # If pool is populated reset the flag since it is set to
            # false during initialisation
            if hasattr(self._flow_proposal, 'resume_populated'):
                if (self._flow_proposal.resume_populated and
                        self._flow_proposal.indices):
                    self._flow_proposal.populated = True
                    logger.info('Resumed with populated pool')

            self.resumed = False

    def finalise(self):
        """
        Finalise things after sampling
        """
        logger.info('Finalising')
        for i, p in enumerate(self.live_points):
            self.state.increment(p['logL'], nlive=self.nlive-i)
            self.nested_samples.append(p)

        # Refine evidence estimate
        self.update_state(force=True)
        self.state.finalise()
        # output the chain and evidence
        self.finalised = True

    def nested_sampling_loop(self):
        """
        Main nested sampling loop
        """
        self.sampling_start_time = datetime.datetime.now()
        if not self.initialised:
            self.initialise(live_points=True)

        if self.prior_sampling:
            self.nested_samples = self.live_points.copy()
            return self.nested_samples

        self.check_resume()

        if self.iteration:
            self.update_state()

        logger.critical('Starting nested sampling loop')

        while self.condition > self.tolerance:

            self.check_state()

            self.consume_sample()

            self.update_state()

            if self.iteration >= self.max_iteration:
                break

        if self.proposal.pool is not None:
            self.proposal.close_pool()

        # final adjustments
        # avoid repeating final adjustments if resuming a completed run.
        if not self.finalised and (self.condition <= self.tolerance):
            self.finalise()

        logger.critical(f'Final evidence: {self.state.logZ:.3f} +/- '
                        f'{np.sqrt(self.state.info[-1] / self.nlive):.3f}')
        logger.critical('Information: {0:.2f}'.format(self.state.info[-1]))

        self.check_insertion_indices(rolling=False)

        # This includes updating the total sampling time
        self.checkpoint(periodic=True)

        logger.info(f'Total sampling time: {self.sampling_time}')
        logger.info(f'Total training time: {self.training_time}')
        logger.info(f'Total population time: {self.proposal_population_time}')
        logger.info(
            f'Total likelihood evaluations: {self.likelihood_calls:3d}')
        if self.proposal.logl_eval_time.total_seconds():
            logger.info(
                'Time spent evaluating likelihood: '
                f'{self.likelihood_evaluation_time}')

        return self.state.logZ, np.array(self.nested_samples)

    @classmethod
    def resume(cls, filename, model, flow_config={}, weights_file=None):
        """
        Resumes the interrupted state from a checkpoint pickle file.

        Parameters
        ----------
        filename : str
            Pickle pickle to resume from
        model : :obj:`nessai.model.Model`
            User-defined model
        flow_config : dict, optional
            Dictionary for configuring the flow
        weights_file : str, optional
            Weights files to use in place of the weights file stored in the
            pickle file.

        Returns
        -------
        obj
            Instance of NestedSampler
        """
        logger.critical('Resuming NestedSampler from ' + filename)
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        model.likelihood_evaluations += obj.likelihood_evaluations[-1]
        obj.model = model
        obj._uninformed_proposal.resume(model)
        obj._flow_proposal.resume(model, flow_config, weights_file)

        obj.resumed = True
        return obj

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
