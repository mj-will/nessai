import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import ksone
import torch

from .posterior import logsubexp, log_integrate_log_trap
from .livepoint import live_points_to_array, get_dtype


logger = logging.getLogger(__name__)


class _NSintegralState(object):
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
    self.iteration=0
    self.logZ=-np.inf
    self.oldZ=-np.inf
    self.logw=0
    self.info=0
    # Start with a dummy sample enclosing the whole prior
    self.logLs=[-np.inf] # Likelihoods sampled
    self.log_vols=[0.0] # Volumes enclosed by contours
  def increment(self, logL, nlive=None):
    """
    Increment the state of the evidence integrator
    Simply uses rectangle rule for initial estimate
    """
    if(logL<=self.logLs[-1]):
      logger.warning('NS integrator received non-monotonic logL. {0:.5f} -> {1:.5f}'.format(self.logLs[-1],logL))
    if nlive is None:
      nlive = self.nlive
    oldZ = self.logZ
    logt=-1.0/nlive
    Wt = self.logw + logL + logsubexp(0,logt)
    self.logZ = np.logaddexp(self.logZ,Wt)
    # Update information estimate
    if np.isfinite(oldZ) and np.isfinite(self.logZ) and np.isfinite(logL):
        self.info = np.exp(Wt - self.logZ)*logL + np.exp(oldZ - self.logZ)*(self.info + oldZ) - self.logZ
        if np.isnan(self.info):
            self.info=0

    # Update history
    self.logw += logt
    self.iteration += 1
    self.logLs.append(logL)
    self.log_vols.append(self.logw)
  def finalise(self):
    """
    Compute the final evidence with more accurate integrator
    Call at end of sampling run to refine estimate
    """
    from scipy import integrate
    # Trapezoidal rule
    self.logZ = log_integrate_log_trap(np.array(self.logLs),np.array(self.log_vols))
    return self.logZ
  def plot(self,filename):
    """
    Plot the logX vs logL
    """
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pyplot as plt
    fig=plt.figure()
    plt.plot(self.log_vols,self.logLs)
    plt.title('{0} iterations. logZ={1:.2f} H={2:.2f} bits'.format(self.iteration,self.logZ,self.info*np.log2(np.e)))
    plt.grid(which='both')
    plt.xlabel('log prior_volume')
    plt.ylabel('log likelihood')
    plt.xlim([self.log_vols[-1],self.log_vols[0]])
    plt.savefig(filename)
    logger.info('Saved nested sampling plot as {0}'.format(filename))


class NestedSampler:
    """
    Nested Sampler class.
    Initialisation arguments:

    model: :obj:`cpnest.Model` user defined model

    manager: `multiprocessing` manager instance which controls
        the shared objects.
        Default: None

    nlive: int
        number of live points to be used for the integration
        Default: 1024

    output: string
        folder where the output will be stored
        Default: None

    verbose: int
        0: Nothing
        1: display information on screen
        2: (1) + diagnostic plots
        Default: 1

    seed: int
        seed for the initialisation of the pseudorandom chain
        Default: 1234

    prior_sampling: boolean
        produce nlive samples from the prior.
        Default: False

    stopping: float
        Stop when remaining samples wouldn't change logZ estimate by this much.
        Deafult: 0.1

    n_periodic_checkpoint: int
        checkpoint the sampler every n_periodic_checkpoint iterations
        Default: None (disabled)

    """

    def __init__(self, model, nlive=1000, output=None, prior_sampling=False,
                 stopping=0.1, flow_class=None, flow_config={}, train_on_empty=True,
                 cooldown=100, memory=False, acceptance_threshold=0.05, analytic_priors = False,
                 maximum_uninformed=1000, training_frequency=1000, uninformed_proposal=None,
                 reset_weights=False, checkpointing=True, resume_file=None,
                 uninformed_proposal_kwargs={}, seed=1234, plot=True,
                 **kwargs):
        """
        Initialise all necessary arguments and
        variables for the algorithm
        """
        logger.info('Initialising nested sampler')
        self.model          = model
        self.prior_sampling = prior_sampling
        self.setup_random_seed(seed)
        self.verbose        = 3
        self.acceptance     = 1.0
        self.accepted       = 0
        self.rejected       = 1
        self.last_updated = 0
        self.iteration      = 0
        self.block_acceptance = 1.
        self.mean_block_acceptance = 1.
        self.block_iteration = 0
        self.nlive          = nlive
        self.live_points = None
        self.insertion_indices = []
        self.rolling_p      = []
        self.checkpointing = checkpointing
        self.tolerance      = stopping
        self.condition      = np.inf
        self.worst          = 0
        self.logLmin        = -np.inf
        self.logLmax        = -np.inf
        self.nested_samples = []
        self.logZ           = None
        self.state          = _NSintegralState(self.nlive)
        self.output_file, self.evidence_file, self.resume_file = \
                self.setup_output(output, resume_file)
        self.output = output
        header              = open(os.path.join(output,'header.txt'),'w')
        header.write('\t'.join(self.model.names))
        header.write('\tlogL\n')

        header.close()

        self.plot = plot

        self.acceptance_threshold = acceptance_threshold
        self.train_on_empty = train_on_empty
        self.cooldown = cooldown
        self.memory = memory
        self.reset_weights = float(reset_weights)
        if training_frequency in [None, 'inf', 'None']:
            logger.warning('Proposal will only train when empty')
            self.training_frequency = np.inf
        else:
            self.training_frequency = training_frequency

        self.max_count = 0

        self.initialised    = False

        logger.info(f'Parsing kwargs to FlowProposal: {kwargs}')
        proposal_output = self.output + '/proposal/'
        if flow_class is not None:
            self._flow_proposal = flow_class(model, flow_config=flow_config,
                    output=propoal_output, plot=plot, **kwargs)
        else:
            from .proposal import FlowProposal
            self._flow_proposal = FlowProposal(model, flow_config=flow_config,
                    output=proposal_output, plot=plot, **kwargs)


        # Uninformed proposal is used for prior sampling
        # If maximum uninformed is greater than 0, the it will be used for
        # another n interation or until it becomes inefficient
        if uninformed_proposal is not None:
            self._uninformed_proposal = unfinformed_proposa(model,
                    **uninformed_proposal_kwargs)
        else:
            if analytic_priors:
                from .proposal import AnalyticProposal
                self._uninformed_proposal = AnalyticProposal(model,
                        **uninformed_proposal_kwargs)
            else:
                from .proposal import RejectionProposal
                self._uninformed_proposal = RejectionProposal(model, poolsize=self.nlive,
                        **uninformed_proposal_kwargs)

        if not maximum_uninformed or maximum_uninformed is None:
            self.uninformed_sampling = False
            self.maximum_uninformed = 0
        else:
            self.uninformed_sampling = True
            self.maximum_uninformed = maximum_uninformed


    def setup_output(self, output, resume_file=None):
        """
        Set up the output folder

        -----------
        Parameters:
        output: string
            folder where the results will be stored
        -----------
        Returns:
            output_file, evidence_file, resume_file: tuple
                output_file:   file where the nested samples will be written
                evidence_file: file where the evidence will be written
                resume_file:   file used for checkpointing the algorithm
        """
        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)
        chain_filename = "chain_"+str(self.nlive)+"_"+str(self.seed)+".txt"
        output_file   = os.path.join(output,chain_filename)
        evidence_file = os.path.join(output,chain_filename+"_evidence.txt")
        if resume_file is None:
            resume_file  = os.path.join(output,"nested_sampler_resume.pkl")
        else:
            resume_file  = os.path.join(output, resume_file)

        return output_file, evidence_file, resume_file


    def write_nested_samples_to_file(self):
        """
        Writes the nested samples to a text file
        """
        np.savetxt(self.output_file, self.nested_samples,
                header='\t'.join(self.live_points.dtype.names))

    def write_evidence_to_file(self):
        """
        Write the evidence logZ and maximum likelihood to the evidence_file
        """
        with open(self.evidence_file,"w") as f:
            f.write('{0:.5f} {1:.5f} {2:.5f}\n'.format(self.state.logZ, self.logLmax, self.state.info))

    def setup_random_seed(self,seed):
        """
        initialise the random seed
        """
        self.seed = seed
        np.random.seed(seed=self.seed)
        torch.manual_seed(self.seed)

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

        analytic_cdf = np.arange(self.nlive + 1) / self.nlive
        counts, _ = np.histogram(indices, bins=np.arange(self.nlive + 1))
        cdf = np.cumsum(counts) / len(indices)
        gaps = np.column_stack([cdf - analytic_cdf[:self.nlive],
            analytic_cdf[1:] - cdf])
        D = np.max(gaps)
        p = ksone.sf(D, self.nlive)

        if rolling:
            logger.warning('Rolling KS test: D={0:.3}, p-value={1:.3}'.format(D, p))
            self.rolling_p.append(p)
        else:
            logger.warning('Final KS test: D={0:.3}, p-value={1:.3}'.format(D, p))

        if filename is not None:
            np.savetxt(os.path.join(
                self.output, filename),
                self.insertion_indices,
                newline='\n',delimiter=' ')


    def yield_sample(self, oldparam):
        """
        Draw points and applying rejection sampling
        """
        while True:
            counter = 0
            while True:
                counter += 1
                newparam = self.proposal.draw(oldparam.copy())
                newparam['logP'] = self.model.log_prior(newparam)

                if newparam['logP'] != -np.inf:
                    newparam['logL'] = self.model.log_likelihood(newparam)
                    if newparam['logL'] > self.logLmin:
                        self.logLmax= max(self.logLmax, newparam['logL'])
                        oldparam = newparam.copy()
                        break
                if (1 / counter) < self.acceptance_threshold:
                    self.max_count += 1
                    break
            yield counter, oldparam

    def insert_live_point(self, live_point):
        """
        Insert a live point
        """
        # This is the index including the current worst point, so final index
        # is one less, otherwise index=0 would never be possible
        index = np.searchsorted(self.live_points['logL'], live_point['logL'])
        # Concatentate is complied C code, so it is much faster than np.insert
        # it also allows for simultaneous removal of the worst point
        # and insertion of the new live point
        self.live_points = np.concatenate([self.live_points[1:index], [live_point],
            self.live_points[index:]])
        return index - 1

    def consume_sample(self):
        """
        Replace a sample for single thread
        """
        worst = self.live_points[0]    # Should this be a copy?
        self.logLmin = np.float128(worst['logL'])
        self.state.increment(worst['logL'])
        self.nested_samples.append(worst)

        self.condition = np.logaddexp(self.state.logZ,
                self.logLmax - self.iteration/(float(self.nlive))) - self.state.logZ

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
                break
            else:
                self.rejected += 1
                self.check_state(rejected=True)
                # if retrained in whilst proposing a sample then update the
                # iteration count since will be zero otherwise
                if not self.block_iteration:
                    self.block_iteration += 1

        self.acceptance = self.accepted / (self.accepted + self.rejected)
        self.mean_block_acceptance = self.block_acceptance / self.block_iteration
        if self.verbose:
            logger.info(("{0:5d}: n: {1:3d} NS_acc: {2:.3f} b_acc: {3:.3f} "
            "sub_acc: {4:.3f} H: {5:.2f} logL: {6:.5f} --> {7:.5f} dZ: {8:.3f} "
            "logZ: {9:.3f} logLmax: {10:.2f}").format(self.iteration, count,
                self.acceptance, self.mean_block_acceptance, 1 / count, self.state.info,
                self.logLmin, proposed['logL'], self.condition, self.state.logZ,
                self.logLmax))

    def populate_live_points(self):
        """
        Initialise the pool of `cpnest.parameter.LivePoint` by
        sampling them from the `cpnest.model.log_prior` distribution
        """
        # send all live points to the samplers for start
        i = 0
        live_points = np.array([], dtype=get_dtype(self.model.names, 'f8'))
        with tqdm(total=self.nlive, disable= not self.verbose, desc='Drawing live points') as pbar:
            while i < self.nlive:
                while i < self.nlive:
                    count, live_point = next(self.yield_sample(self.model.new_point()))
                    if np.isnan(live_point['logL']):
                        logger.warning("Likelihood function returned NaN for live_points " + str(live_points[i]))
                        logger.warning("You may want to check your likelihood function")
                    if live_point['logP']!=-np.inf and live_point['logL'] != -np.inf:
                        i+=1
                        live_points = np.concatenate([live_points, [live_point]])
                        pbar.update()
                        break

        self.live_points= np.sort(live_points, order='logL')

    def initialise(self, live_points=True):
        """
        Initialise the nested sampler
        """
        flags = [False] * 3
        if not self._flow_proposal.intialised:
            self._flow_proposal.initialise()
            flags[0] = True

        if not self._uninformed_proposal.intialised:
            self._uninformed_proposal.initialise()
            flags[1] = True

        if self.iteration < self.maximum_uninformed:
            self.proposal = self._uninformed_proposal
        else:
            self.proposal = self._flow_proposal

        if live_points and self.live_points is None:
            self.populate_live_points()
            flags[2] = True

        if all(flags):
            self.initialised = True

    def check_state(self, force=False, rejected=False):
        """
        Check if state should be updated prior to drawing a new sample

        Force will overide the cooldown mechanism, rejected will not
        """
        if self.uninformed_sampling:
            if (rejected and self.acceptance < self.acceptance_threshold) or \
                    self.iteration >= self.maximum_uninformed:
                logger.warning('Switching to FlowProposal')
                self.proposal = self._flow_proposal
                self.uninformed_sampling = False
        # Should the proposal be trained
        train = False
        # General overide
        if force:
            train = True
            logger.debug('Training flow (force)')
        elif self.mean_block_acceptance < self.acceptance_threshold:
            train = True
            logger.debug('Training flow (acceptance)')
        elif rejected and self.mean_block_acceptance < self.acceptance_threshold:
            logger.debug('Training flow (rejected + acceptance)')
            train = True
        elif not self.proposal.populated:
            if self.train_on_empty:
                train = True
                logger.debug('Training flow (proposal empty)')

        elif not (self.iteration - self.last_updated) % self.training_frequency:
            if not self.uninformed_sampling:
                train = True
                logger.debug('Training flow (iteration)')

        if train:
            if self.iteration - self.last_updated < self.cooldown and not force:
                logger.debug('Not training, still cooling down!')
            else:
                if self.reset_weights and not (self.proposal.training_count % self.reset_weights):
                    self.proposal.reset_model_weights()
                training_data = self.live_points.copy()
                if self.memory:
                    if len(self.nested_samples):
                        if len(self.nested_samples) >= self.memory:
                            training_data = np.concatenate([training_data, self.nested_samples[-self.memory].copy()])
                self.proposal.train(training_data, plot=self.plot)
                self.last_updated = self.iteration
                self.block_iteration = 0
                self.block_acceptance = 1.0
                if self.checkpointing:
                    self.checkpoint()

    def update_state(self):
        """
        Update state after replacing a live point
        """
        if not (self.iteration % self.nlive):
            self.check_insertion_indices()

    def checkpoint(self):
        """
        Checkpoint its internal state
        """
        logger.critical('Checkpointing nested sampling')
        with open(self.resume_file,"wb") as f:
            pickle.dump(self, f)

    def nested_sampling_loop(self, save=True):
        """
        main nested sampling loop
        """
        if not self.initialised:
            self.initialise()

        if self.prior_sampling:
            for i in range(self.nlive):
                self.nested_samples = self.params.copy()
                #self.nested_samples.append(self.params[i])
            if save:
                self.write_nested_samples_to_file()
                self.write_evidence_to_file()
            logger.warning("Nested Sampling process {0!s}, exiting".format(os.getpid()))
            return 0

        while self.condition > self.tolerance:

            self.check_state()

            self.consume_sample()

            self.update_state()

        # final adjustments
        for i, p in enumerate(self.live_points):
            self.state.increment(p['logL'], nlive=self.nlive-i)
            self.nested_samples.append(p)
        self.nested_samples = np.array(self.nested_samples)

        # Refine evidence estimate
        self.state.finalise()
        self.logZ = self.state.logZ
        # output the chain and evidence
        if save:
            self.write_nested_samples_to_file()
            self.write_evidence_to_file()

        logger.critical('Final evidence: {0:0.2f}'.format(self.state.logZ))
        logger.critical('Information: {0:.2f}'.format(self.state.info))

        return self.state.logZ, self.nested_samples

    @classmethod
    def resume(cls, filename, model):
        """
        Resumes the interrupted state from a
        checkpoint pickle file.
        """
        logger.critical('Resuming NestedSampler from ' + filename)
        with open(filename,"rb") as f:
            obj = pickle.load(f)
        obj.model = model
        obj._flow_proposal.flow.reload_weights()
        obj._flow_proposal.model = model
        obj._uninformed_proposal.model = model
        return obj

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
