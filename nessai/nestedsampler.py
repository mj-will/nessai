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
from .posterior import log_integrate_log_trap
from .proposal import FlowProposal
from .utils import (
    safe_file_dump,
    compute_indices_ks_test,
    )

sns.set()
sns.set_style('ticks')

logger = logging.getLogger(__name__)


class _NSintegralState:
    """
    Stores the state of the nested sampling integrator
    """
    def __init__(self, nlive):
        self.nlive = nlive
        self.reset()

    def reset(self):
        """
        Reset the sampler to its initial state at logZ = -infinity
        """
        self.logZ = -np.inf
        self.oldZ = -np.inf
        self.logw = 0
        self.info = [0.]
        # Start with a dummy sample enclosing the whole prior
        self.logLs = [-np.inf]   # Likelihoods sampled
        self.log_vols = [0.0]    # Volumes enclosed by contours
        self.gradients = [0]

    def increment(self, logL, nlive=None):
        """
        Increment the state of the evidence integrator
        Simply uses rectangle rule for initial estimate
        """
        if(logL <= self.logLs[-1]):
            logger.warning('NS integrator received non-monotonic logL.'
                           f'{self.logLs[-1]:.5f} -> {logL:.5f}')
        if nlive is None:
            nlive = self.nlive
        oldZ = self.logZ
        logt = - 1.0 / nlive
        Wt = self.logw + logL + np.log1p(-np.exp(logt))
        self.logZ = np.logaddexp(self.logZ, Wt)
        # Update information estimate
        if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
            info = np.exp(Wt - self.logZ) * logL \
                  + np.exp(oldZ - self.logZ) \
                  * (self.info[-1] + oldZ) \
                  - self.logZ
            if np.isnan(info):
                info = 0
            self.info.append(info)

        # Update history
        self.logw += logt
        self.logLs.append(logL)
        self.log_vols.append(self.logw)
        self.gradients.append((self.logLs[-1] - self.logLs[-2])
                              / (self.log_vols[-1] - self.log_vols[-2]))

    def finalise(self):
        """
        Compute the final evidence with more accurate integrator
        Call at end of sampling run to refine estimate
        """
        # Trapezoidal rule
        self.logZ = log_integrate_log_trap(np.array(self.logLs),
                                           np.array(self.log_vols))
        return self.logZ

    def plot(self, filename):
        """
        Plot the logX vs logL
        """
        fig = plt.figure()
        plt.plot(self.log_vols, self.logLs)
        plt.title(f'logZ={self.logZ:.2f}'
                  f'H={self.info[-1] * np.log2(np.e):.2f} bits')
        plt.grid(which='both')
        plt.xlabel('log prior_volume')
        plt.ylabel('log likelihood')
        plt.xlim([self.log_vols[-1], self.log_vols[0]])
        plt.yscale('symlog')
        fig.savefig(filename)
        logger.info('Saved nested sampling plot as {0}'.format(filename))


class NestedSampler:
    """
    Nested Sampler class.
    Initialisation arguments:

    Parameters
    ----------
    model: :obj:`nessai.Model`
        User defined model
    nlive : int, optional
        Number of live points. Defaults to 1000
    output : str
        Path for output
    stopping : float (0.1)
        Stop when remaining samples wouldn't change logZ estimate by this much.
    max_iteration : int (None)
        Maximum number of iterations to run before force sampler to stop.
        If stopping criteria is met before max. is reached sampler will stop.
    checkpoint : bool (True)
        Boolean to toggle checkpointing, must be enable to resume sampler
    resume_file : str (None)
        If specified sampler will be resumed from this file. Still requieres
        correct model.
    seed : int
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
        Class to use for inintial sampling before training the flow. If
        None RejectionProposal or AnalyticProposal are used depending if
        `analytic_priors` is False or True.
    uninformed_acceptance_threshold : float (None)
        Acceptance threshold for initialing sampling, if acceptance falls
        below this value sampler switches to flow-based proposal. If None
        then value is set to 10 times `acceptance_threshold`
    uninformed_proposal_kwargs : dict, ({})
        Dictionary of keyword argument to parase to the class use for
        the intial sampling when it is initialised.
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
        Boolean to toggle reseting the flow weights whenever re-training.
        If an interger is specified the flow is reset every nth time it is
        trained.
    reset_permuations: bool, int, (False)
        Boolean to toggle reseting the permuation layers in the flow whenever
        re-training. If an interger is specified the flow is reset every nth
        time it is trained.
    reset_acceptance : bool, (True)
        If true use mean acceptance of samples produced with current flow
        as a criteria for retraining
    retrain_acceptance : bool (False)
        Force the flow to be reset if the acceptance falls below the acceptance
        threshold. Requiers `reset_acceptance=True`
    acceptance_threshold : float (0.01)
        Threshold to determine if the flow should be retrained, will not
        retrain if cooldown is not satisfied.
    kwargs :
        Keyword arguments parsed to the flow proposal class
    """

    def __init__(self, model, nlive=1000, output=None,
                 stopping=0.1,
                 max_iteration=None,
                 checkpointing=True,
                 checkpoint_on_training=False,
                 resume_file=None,
                 seed=None,
                 n_pool=None,
                 plot=True,
                 proposal_plots=True,
                 prior_sampling=False,
                 analytic_priors=False,
                 maximum_uninformed=1000,
                 uninformed_proposal=None,
                 uninformed_acceptance_threshold=None,
                 uninformed_proposal_kwargs=None,
                 flow_class=None,
                 flow_config=None,
                 training_frequency=None,
                 train_on_empty=True,
                 cooldown=100,
                 memory=False,
                 reset_weights=False,
                 reset_permutations=False,
                 retrain_acceptance=True,
                 reset_acceptance=False,
                 acceptance_threshold=0.01,
                 **kwargs):

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
        self.state = _NSintegralState(self.nlive)
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

        if max_iteration is None:
            self.max_iteration = np.inf
        else:
            self.max_iteration = max_iteration

        self.acceptance_threshold = acceptance_threshold

        self.train_on_empty = train_on_empty
        self.cooldown = cooldown
        self.memory = memory

        self.configure_flow_reset(reset_weights, reset_permutations)

        if training_frequency in [None, 'inf', 'None']:
            logger.warning('Proposal will only train when empty')
            self.training_frequency = np.inf
        else:
            self.training_frequency = training_frequency

        self.max_count = 0

        self.initialised = False

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
        # another n interation or until it becomes inefficient

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
        return self.sampling_time \
                + (datetime.datetime.now() - self.sampling_start_time)

    @property
    def last_updated(self):
        """Last time the normalising flow was retraining"""
        if self.training_iterations:
            return self.training_iterations[-1]
        else:
            return 0

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
            If None, no max is set. If False uninformed sampling is not used.
        uninformed_acceptance_threshold :  float or None:
            Threshold to use for uninformed proposal, once reached proposal
            method will switch. If None acceptance_threshold is used if
            greater than 0.1 else 10 x acceptance_threshold is used.
        kwargs
            Kwargs are parsed to init method for uninformed proposal class
        """
        if maximum_uninformed is None:
            self.uninformed_sampling = True
            self.maximum_uninformed = np.inf
        elif not maximum_uninformed:
            self.uninformed_sampling = False
            self.maximum_uninformed = 0
        else:
            self.uninformed_sampling = True
            self.maximum_uninformed = maximum_uninformed

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
        flow_class : None or obj
            Class to use for proposal. If None FlowProposal is used.
        flow_config : dict
            Configuration dictionary parsed to the class
        proposal_plots : bool or str
            Configuration of plottinmg in proposal class
        **kwargs :
            Kwargs parsed to init function
        """
        proposal_output = self.output + '/proposal/'

        if not self.plot:
            proposal_plots = False

        if flow_class is not None:
            if isinstance(flow_class, str):
                if flow_class == 'GWFlowProposal':
                    from .gw.proposal import GWFlowProposal as flow_class
                elif flow_class == 'LegacyGWFlowProposal':
                    from .gw.proposal import LegacyGWFlowProposal as flow_class
                elif flow_class == 'FlowProposal':
                    flow_class = FlowProposal
                else:
                    raise RuntimeError(f'Unknown flow class: {flow_class}')
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
        if isinstance(reset_weights, (int, float)):
            self.reset_weights = float(reset_weights)
        else:
            raise RuntimeError
        if isinstance(reset_permutations, (int, float)):
            self.reset_permutations = float(reset_permutations)
        else:
            raise RuntimeError

    def check_insertion_indices(self, rolling=True, filename=None):
        """
        Checking the distibution of the insertion indices either during
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
                        logger.warning('Likelihood function returned NaN for '
                                       'live_points ' + str(live_points[i]))
                        logger.warning(
                            'You may want to check your likelihood function')
                    if (live_point['logP'] != -np.inf and
                            live_point['logL'] != -np.inf):
                        live_points[i] = live_point
                        i += 1
                        pbar.update()
                        break

        self.live_points = np.sort(live_points, order='logL')
        if self.store_live_points:
            np.savetxt(self.live_points_dir + '/intial_live_points.dat',
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

        if self.iteration < self.maximum_uninformed:
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

    @property
    def mean_acceptance(self):
        """
        Mean acceptance of the last nlive // 10 points
        """
        return np.mean(self.acceptance_history)

    def check_proposal_switch(self):
        """
        Check if the proposal should be switch from uninformed to
        flowproposal given the current state.

        Returns
        -------
        bool
            Flag to indicated if proposal was switched
        """
        if ((self.mean_acceptance < self.uninformed_acceptance_threshold)
                or (self.iteration >= self.maximum_uninformed)):
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
        Check if the normalising flow model should be reset
        """
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
        Try to trin the proposal. Proposal will not train if cooldown is not
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

        Force will overide the cooldown mechanism.
        """

        if self.uninformed_sampling:
            if self.check_proposal_switch():
                force = True
            else:
                return
        # General overide
        train = False
        if force:
            train = True
            logger.debug('Training flow (force)')
        elif not train:
            train, force = self.check_training()

        if train or force:
            self.train_proposal(force=force)

    def plot_state(self):
        """
        Produce plots with the current state of the nested sampling run.
        Plots are saved to the output directory specifed at initialisation.
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

        g = np.min([len(self.state.gradients), self.iteration])
        ax[1].plot(np.arange(g), np.abs(self.state.gradients[:g]),
                   c=colours[0], label='Gradient')
        ax[1].set_ylabel(r'$|d\log L/d \log X|$')
        ax[1].set_yscale('log')

        ax[2].plot(it, self.likelihood_evaluations, c=colours[0], ls=ls[0],
                   label='Evalutions')
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

        fig.savefig(f'{self.output}/state.png')

    def plot_trace(self):
        """
        Make trace plots for the nested samples
        """
        if self.nested_samples:
            plot_trace(self.state.log_vols[1:], self.nested_samples,
                       filename=f'{self.output}/trace.png')
        else:
            logger.warning('Could not produce trace plot. No nested samples!')

    def update_state(self, force=False):
        """
        Update state after replacing a live point
        """
        # Check if acceptance is not None, this indicates the proposal
        # was populated
        if (pa := self.proposal.population_acceptance) is not None:
            self.population_acceptance.append(pa)
            self.population_radii.append(self.proposal.r)
            self.population_iterations.append(self.iteration)

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
                self.plot_state()
                self.plot_trace()

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
            or forced by a signal
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
            for i in range(self.nlive):
                self.nested_samples = self.params.copy()
            return

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
