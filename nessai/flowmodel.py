# -*- coding: utf-8 -*-
"""
Object and functions to handle training the normalising flow.
"""
import copy
import glob
import json
import logging
import numpy as np
import os
import shutil
import torch
from torch.nn.utils import clip_grad_norm_

from .flows import (
    configure_model,
    set_affine_parameters,
    reset_weights,
    reset_permutations,
)
from .plot import plot_loss
from .utils import NessaiJSONEncoder, compute_minimum_distances

logger = logging.getLogger(__name__)


def update_config(d):
    """
    Update the configuration dictionary to include the defaults.

    Notes
    -----
    The default configuration is::

        lr=0.001
        annealing=False
        batch_size=100
        val_size=0.1
        max_epochs=500
        patience=20
        noise_scale=0.0,
        use_dataloader=False,
        optimiser='adam',
        optimiser_kwargs={}
        model_config=default_model

    where ``default model`` is::

        n_neurons=32
        n_blocks=4
        n_layers=2
        ftype='RealNVP'
        device_tag='cpu',
        flow=None,
        inference_device_tag=None,
        distribution=None,
        distribution_kwargs=None,
        kwargs={batch_norm_between_layers=True, linear_transform='lu'}

    The kwargs can contain any additional keyword arguments that are specific
    to the type of flow being used.

    Parameters
    ----------
    d : dict
        Dictionary with configuration

    Returns
    -------
    dict
        Dictionary with updated default configuration
    """
    default_model = dict(
        n_inputs=None,
        n_neurons=None,
        n_blocks=4,
        n_layers=2,
        ftype='RealNVP',
        device_tag='cpu',
        flow=None,
        inference_device_tag=None,
        distribution=None,
        distribution_kwargs=None,
        kwargs=dict(
            batch_norm_between_layers=True,
            linear_transform='lu'
        )
    )

    default = dict(
        lr=0.001,
        annealing=False,
        clip_grad_norm=5,
        batch_size=1000,
        val_size=0.1,
        max_epochs=500,
        patience=20,
        noise_scale=0.0,
        use_dataloader=False,
        optimiser='adamw',
        optimiser_kwargs={}
    )

    if d is None:
        default['model_config'] = default_model
    else:
        if not isinstance(d, dict):
            raise TypeError('Must pass a dictionary to update the default '
                            'trainer settings')
        else:
            default.update(d)
            default_model.update(d.get('model_config', {}))
            if default_model['n_neurons'] is None:
                if default_model['n_inputs'] is not None:
                    default_model['n_neurons'] = 2 * default_model['n_inputs']
                else:
                    default_model['n_neurons'] = 8

            default['model_config'] = default_model

    if (
        not isinstance(default['noise_scale'], float) and
        not default['noise_scale'] == 'adaptive'
    ):
        raise ValueError(
            "noise_scale must be a float or 'adaptive'. "
            f"Received: {default['noise_scale']}"
        )

    return default


class FlowModel:
    """
    Object that contains the normalsing flows and handles training and data
    pre-processing.

    Does NOT use structured arrays for live points, \
            :obj:`~nessai.proposal.base.Proposal`
    object should act as the interface between structured used by the sampler
    and unstructured arrays of live points used for training.

    Parameters
    ----------
    config : dict, optional
        Configuration used for the normalising flow. If None, default values
        are used.
    output : str, optional
        Path for output, this includes weights files and the loss plot.
    """
    model_config = None

    def __init__(self, config=None, output='./'):
        self.model = None
        self.initialised = False
        self.output = output
        os.makedirs(self.output, exist_ok=True)
        self.setup_from_input_dict(config)
        self.weights_file = None

        self.device = None
        self.inference_device = None
        self.use_dataloader = False
        self.__has_affine = None

    @property
    def _has_affine(self):
        """Check if the flow contains a pointwise affine transform."""
        if self.__has_affine is None:
            if self.model is None:
                return None
            from nflows.transforms import PointwiseAffineTransform
            for module in self.model.modules():
                if isinstance(module, PointwiseAffineTransform):
                    self.__has_affine = True
                    break
            if self.__has_affine is None:
                self.__has_affine = False
        return self.__has_affine

    def save_input(self, config, output_file=None):
        """
        Save the dictionary used as an inputs as a JSON file in the output
        directory.

        Parameters
        ----------
        config : dict
            Dictionary to save.
        output_file : str, optional
            File to save the config to.
        """
        config = copy.deepcopy(config)
        if output_file is None:
            output_file = self.output + "flow_config.json"
        for k, v in list(config.items()):
            if type(v) == np.ndarray:
                config[k] = np.array_str(config[k])
        for k, v in list(config['model_config'].items()):
            if type(v) == np.ndarray:
                config['model_config'][k] = \
                    np.array_str(config['model_config'][k])

        if 'flow' in config['model_config']:
            config['model_config']['flow'] = \
                str(config['model_config']['flow'])

        with open(output_file, "w") as f:
            json.dump(config, f, indent=4, cls=NessaiJSONEncoder)

    def setup_from_input_dict(self, config):
        """
        Setup the trainer from a dictionary, all keys in the dictionary are
        added as methods to the object. Input is automatically saved.

        Parameters
        ----------
        config : dict
            Dictionary with parameters that are used to update the defaults.
        """
        config = update_config(config)
        logger.debug(f'Flow configuration: {config}')
        for key, value in config.items():
            setattr(self, key, value)
        self.save_input(config)

    def update_mask(self):
        """Method to update the ask upon calling ``initialise``

        By default the mask is left unchanged.
        """
        pass

    def get_optimiser(self, optimiser='adam', **kwargs):
        """
        Get the optimiser and ensure it is always correctly initialised.

        Returns
        -------
        :obj:`torch.optim.Adam`
            Instance of the Adam optimiser from torch.optim
        """
        optimisers = {
            'adam': (torch.optim.Adam, {'weight_decay': 1e-6}),
            'adamw': (torch.optim.AdamW, {}),
            'sgd': (torch.optim.SGD, {})
        }
        if self.model is None:
            raise RuntimeError('Cannot initialise optimiser before model')
        optim, default_kwargs = optimisers.get(optimiser.lower())
        default_kwargs.update(kwargs)
        return optim(self.model.parameters(), lr=self.lr, **default_kwargs)

    def initialise(self):
        """
        Initialise the model and optimiser.

        This includes:

            - Updating the model configuration
            - Initialising the normalising flow
            - Initialising the optimiser
            - Configuring the inference device
        """
        self.update_mask()
        self.model, self.device = configure_model(self.model_config)
        logger.debug(f'Training device: {self.device}')
        self.inference_device = torch.device(
            self.model_config.get('inference_device_tag', self.device)
            or self.device
        )
        logger.debug(f'Inference device: {self.inference_device}')

        self._optimiser = self.get_optimiser(
            self.optimiser, **self.optimiser_kwargs)
        self.initialised = True

    def move_to(self, device, update_default=False):
        """Move the flow to a different device.

        Parameters
        ----------
        device : str
            Device to move flow to.
        update_default : bool, optional
            If True, the default device for the flow (and data) will be
            updated.
        """
        device = torch.device(device)
        self.model.to(device)
        if update_default:
            self.device = device

    @staticmethod
    def check_batch_size(x, batch_size, min_fraction=0.1):
        """Check that the batch size is valid.

        Tries to ensure that the last batch is at least a minimum fraction of
         the size of the batch size.
        """
        logger.debug('Checking batch size')
        if batch_size == 1:
            raise ValueError('Cannot use a batch size of 1!')
        min_batch_size = int(min_fraction * batch_size)
        final_batch_size = len(x) % batch_size
        if final_batch_size and (final_batch_size < min_batch_size):
            logger.debug(
                'Adjusting batch size to ensure final batch has at least '
                f'{min_batch_size} samples in it.'
            )
            while True:
                batch_size -= 1
                final_batch_size = len(x) % batch_size
                if batch_size < 2:
                    raise RuntimeError('Could not find a valid batch size')
                elif (
                    (final_batch_size == 0)
                    or (final_batch_size >= min_batch_size)
                ):
                    break
                elif (batch_size <= min_batch_size) and final_batch_size > 1:
                    logger.warning(
                        f'Batch size is less than {min_batch_size} but valid. '
                        f'Setting batch size to: {batch_size}'
                    )
                    break
        logger.debug(f'Using valid batch size of: {batch_size}')
        return batch_size

    def prep_data(
        self,
        samples,
        val_size,
        batch_size,
        weights=None,
        use_dataloader=False
    ):
        """
        Prep data and return dataloaders for training

        Parameters
        ----------
        samples : array_like
            Array of samples to split in to training and validation.
        val_size : float
            Float between 0 and 1 that defines the fraction of data used for
            validation.
        batch_size : int
            Batch size used when constructing dataloaders.
        use_dataloader : bool, optional
            If True data is returned in a dataloader else a tensor is returned.
        weights : array_like, optional
            Array of weights for each sample, weights will used when computing
            the loss.

        Returns
        -------
        train_data, val_data :
            Training and validation data as either a tensor or dataloader
        """
        if not self.initialised:
            self.initialise()

        if not np.isfinite(samples).all():
            raise ValueError('Cannot train with non-finite samples!')

        idx = np.random.permutation(samples.shape[0])
        samples = samples[idx]
        if weights is not None:
            if not np.isfinite(weights).all():
                raise ValueError('Weights contain non-finite values!')
            weights = weights[idx]
            use_dataloader = True

        logger.debug("N input samples: {}".format(len(samples)))

        # setup data loading
        n = int((1 - val_size) * samples.shape[0])
        x_train, x_val = samples[:n], samples[n:]
        if weights is not None:
            weights_train = weights[:n]
            weights_val = weights[n:]
        logger.debug(f'{x_train.shape} training samples')
        logger.debug(f'{x_val.shape} validation samples')

        if not type(batch_size) is int:
            if batch_size == 'all' or batch_size is None:
                batch_size = x_train.shape[0]
            else:
                raise RuntimeError(f'Unknown batch size: {batch_size}')

        batch_size = self.check_batch_size(x_train, batch_size)

        dtype = torch.get_default_dtype()
        logger.debug(f'Using dtype {dtype} for tensors')
        if use_dataloader:
            logger.debug('Using dataloaders')
            train_tensors = [torch.from_numpy(x_train).type(dtype)]
            val_tensors = [torch.from_numpy(x_val).type(dtype)]

            if weights is not None:
                train_tensors.append(
                    torch.from_numpy(weights_train).type(dtype)
                )
                val_tensors.append(torch.from_numpy(
                    weights_val).type(dtype)
                )

            train_dataset = torch.utils.data.TensorDataset(*train_tensors)
            train_data = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)

            val_dataset = torch.utils.data.TensorDataset(*val_tensors)
            val_data = torch.utils.data.DataLoader(
                val_dataset, batch_size=x_val.shape[0], shuffle=False)
        else:
            logger.debug('Using tensors')
            train_data = \
                torch.from_numpy(x_train).type(dtype).to(self.device)
            val_data = \
                torch.from_numpy(x_val).type(dtype).to(self.device)
            self._batch_size = batch_size

        return train_data, val_data, batch_size

    def end_iteration(self):
        """Calls any functions that should be applied at the end of the \
            iteration.

        This functions is called after the flow has been updated on all batches
        of data but before the validation step.

        Calls :py:meth:`nessai.flows.base.BaseFlow.end_iteration`
        """
        self.model.end_iteration()

    def _train(
        self,
        train_data,
        noise_scale=0.0,
        is_dataloader=False,
        weighted=False
    ):
        """
        Loop over the data and update the weights

        Parameters
        ----------
        train_data : :obj:`torch.util.data.Dataloader` or :obj:`torch.Tensor`
            Training data. If a tensor is provided, it is split into batches
            using the batch size.
        noise_scale : float, optional
            Scale of Gaussian noise added to data.
        is_dataloader : bool, optional
            Must be True when using a dataloader
        weighted : bool
            If True the weighted KL will be used to compute the loss. Requires
            data to include weights.

        Returns
        -------
        float
            Mean of training loss for each batch.
        """
        model = self.model
        model.train()
        train_loss = 0

        if hasattr(model, 'loss_function'):
            loss_fn = model.loss_function
        elif weighted:
            def loss_fn(data, weights):
                return (
                    -torch.sum(model.log_prob(data) * weights)
                    / torch.sum(weights)
                )
        else:
            def loss_fn(data):
                return -model.log_prob(data).mean()

        if not is_dataloader:
            p = torch.randperm(train_data.shape[0])
            train_data = train_data[p, :].split(self._batch_size)

        n = 0
        for data in train_data:
            if is_dataloader:
                x = data[0].to(self.device)
            else:
                x = data
            if noise_scale:
                x += noise_scale * torch.randn_like(x)
            for param in model.parameters():
                param.grad = None
            if weighted:
                weights = data[1].to(self.device)
                loss = loss_fn(x, weights)
            else:
                loss = loss_fn(x)
            train_loss += loss.item()
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            self._optimiser.step()
            n += 1

        self.end_iteration()

        if self.annealing:
            self.scheduler.step()

        return train_loss / n

    def _validate(self, val_data, is_dataloader=False, weighted=False):
        """
        Loop over the data and get validation loss

        Parameters
        ----------
        val_data : :obj:`torch.util.data.Dataloader` or :obj:`torch.Tensor
            Dataloader with data to validate on
        is_dataloader : bool, optional
            Boolean to indicate if the data is a dataloader
        weighted : bool
            If True the weighted KL will be used to compute the loss. Requires
            data to include weights.

        Returns
        -------
        float
            Mean of training loss for each batch.
        """
        model = self.model
        model.eval()
        val_loss = 0

        if hasattr(model, 'loss_function'):
            loss_fn = model.loss_function
        elif weighted:
            def loss_fn(data, weights):
                return (
                    -torch.sum(model.log_prob(data) * weights)
                    / torch.sum(weights)
                )
        else:
            def loss_fn(data):
                return -model.log_prob(data).mean()

        if is_dataloader:
            n = 0
            for data in val_data:
                if is_dataloader:
                    x = data[0].to(self.device)
                if weighted:
                    weights = data[1].to(self.device)
                    with torch.no_grad():
                        val_loss += loss_fn(x, weights).item()
                else:
                    with torch.no_grad():
                        val_loss += loss_fn(x).item()
                n += 1

            return val_loss / n
        else:
            with torch.no_grad():
                val_loss += loss_fn(val_data).item()
            return val_loss

    def finalise(self):
        """Method to finalise the flow before using it for inference."""
        logger.debug('Finalising model before inference.')
        self.model.finalise()

    def prep_model(self, samples):
        """Prepare the model"""
        if self._has_affine:
            shift = -np.mean(samples, axis=0)
            scale = 1 / np.std(samples, axis=0)
            logger.debug(f'Setting scale and shift to: {scale}, {shift}')
            set_affine_parameters(self.model, scale, shift)

    def train(
        self,
        samples,
        weights=None,
        max_epochs=None,
        patience=None,
        output=None,
        val_size=None,
        plot=True
    ):
        """
        Train the flow on a set of samples.

        Allows for training parameters to specified instead of those
        given in initial config.

        Parameters
        ----------
        samples : ndarray
            Unstructured numpy array containing data to train on
        weights : array_like
            Array of weights to use with the weight KL when computing the loss.
        max_epochs : int, optional
            Maximum number of epochs that is used instead of value
            in the configuration.
        patience : int, optional
            Patience in number of epochs that is used instead of value
            in the configuration.
        val_size : float, optional
           Fraction of the samples to use for validation
        output : str, optional
            Path to output directory that is used instead of the path
            specified when the object is initialised
        plot : bool, optional
            Boolean to enable or disable plotting the loss function
        """
        if not self.initialised:
            logger.info("Initialising")
            self.initialise()

        if output is None:
            output = self.output
        else:
            os.makedirs(output, exist_ok=True)

        if val_size is None:
            val_size = self.val_size

        if self.noise_scale == 'adaptive':
            noise_scale = 0.2 * np.mean(compute_minimum_distances(samples))
            logger.debug(f'Using adaptive scale: {noise_scale:.3f}')
        else:
            noise_scale = self.noise_scale

        self.move_to(self.device)

        weighted = True if weights is not None else False
        use_dataloader = self.use_dataloader or weighted

        train_data, val_data, batch_size = self.prep_data(
            samples,
            val_size=val_size,
            batch_size=self.batch_size,
            weights=weights,
            use_dataloader=use_dataloader
        )

        self.prep_model(samples)

        if max_epochs is None:
            max_epochs = self.max_epochs
        if self.annealing:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self._optimiser, max_epochs)
        if patience is None:
            patience = self.patience
        best_epoch = 0
        best_val_loss = np.inf
        best_model = copy.deepcopy(self.model.state_dict())
        logger.info("Starting training")
        logger.info("Training parameters:")
        logger.info(f"Max. epochs: {max_epochs}")
        logger.info(f"Patience: {patience}")
        logger.info(f'Training with {samples.shape[0]} samples')

        if plot:
            history = dict(loss=[], val_loss=[])

        current_weights_file = output + 'model.pt'
        for epoch in range(1, max_epochs + 1):

            loss = self._train(
                train_data,
                noise_scale=noise_scale,
                is_dataloader=use_dataloader,
                weighted=weighted,
            )
            val_loss = self._validate(
                val_data,
                is_dataloader=use_dataloader,
                weighted=weighted,
            )
            if plot:
                history['loss'].append(loss)
                history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())

            if not epoch % 50:
                logger.info(
                    f'Epoch {epoch}: loss: {loss:.3} val loss: {val_loss:.3}')

            if epoch - best_epoch > patience:
                logger.info(f'Epoch {epoch}: Reached patience')
                break

        logger.debug('Finished training')
        self.model.load_state_dict(best_model)
        self.finalise()

        self.save_weights(current_weights_file)
        self.move_to(self.inference_device)
        self.model.eval()

        if plot:
            plot_loss(epoch, history, filename=output + '/loss.png')

    def save_weights(self, weights_file):
        """
        Save the weights file. If the file already exists move it to
        ``<weights_file>.old`` and then save the file.

        Parameters
        ----------
        weights_file : str
            Path to to file to save weights. Recommended file type is ``.pt``.
        """
        if os.path.exists(weights_file):
            shutil.move(weights_file, weights_file + '.old')

        torch.save(self.model.state_dict(), weights_file)
        self.weights_file = weights_file

    def load_weights(self, weights_file):
        """
        Load weights for the model and initialises the model if it is not
        initialised. The weights_file attribute is also updated.

        Model is loaded in evaluation mode (``model.eval()``)

        Parameters
        ----------
        weights_files : str
            Path to weights file
        """
        # TODO: these two methods are basically the same
        if not self.initialised:
            self.initialise()
        self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()
        self.weights_file = weights_file

    def reload_weights(self, weights_file):
        """
        Tries to the load the weights file and if not, tries to load
        the weights file stored internally.

        Parameters
        ----------
        weights_files : str
            Path to weights file
        """
        if weights_file is None:
            weights_file = self.weights_file
        logger.debug(f'Reloading weights from {weights_file}')
        self.load_weights(weights_file)

    def reset_model(self, weights=True, permutations=False):
        """
        Reset the weights of the model and optimiser.

        Parameters
        ----------
        weights : bool, optional
            If true the model weights are reset.
        permutations : bool, optional
            If true any permutations (linear transforms) are reset.
        """
        if not any([weights, permutations]):
            logger.debug('Nothing to reset')
            return
        if weights and permutations:
            logger.debug('Complete reset of model')
            self.model, self.device = configure_model(self.model_config)
        elif weights:
            self.model.apply(reset_weights)
            logger.debug('Reset weights')
        elif permutations:
            self.model.apply(reset_permutations)
            logger.debug('Reset linear transforms')
        self._optimiser = self.get_optimiser(
            self.optimiser, **self.optimiser_kwargs)
        logger.debug('Resetting optimiser')

    def forward_and_log_prob(self, x):
        """
        Forward pass through the model and return the samples in the latent
        space with their log probabilities

        Parameters
        ----------
        x : ndarray
            Array of samples

        Returns
        -------
        z : ndarray
            Samples in the latent space
        log_prob : ndarray
            Log probabilities for each samples
        """
        x = (
            torch.from_numpy(x).type(torch.get_default_dtype())
            .to(self.model.device)
        )
        self.model.eval()
        with torch.no_grad():
            z, log_prob = self.model.forward_and_log_prob(x)

        z = z.detach().cpu().numpy().astype(np.float64)
        log_prob = log_prob.detach().cpu().numpy().astype(np.float64)
        return z, log_prob

    def log_prob(self, x):
        """Compute the log-probability of a sample.

        Parameters
        ----------
        x : ndarray
            Array of samples in the X-prime space.

        Returns
        -------
        ndarray
            Array of log-probabilities.
        """
        # Should this be the inference device?
        x = (
            torch.from_numpy(x).type(torch.get_default_dtype())
            .to(self.model.device)
        )
        self.model.eval()
        with torch.no_grad():
            log_prob = self.model.log_prob(x)
        log_prob = log_prob.cpu().numpy().astype(np.float64)
        return log_prob

    def sample_and_log_prob(self, N=1, z=None, alt_dist=None):
        """
        Generate samples from samples drawn from the base distribution or
        and alternative distribution from provided latent samples

        Parameters
        ----------
        N : int, optional
            Number of samples to draw if z is not specified
        z : ndarray, optional
            Array of latent samples to map the the data space, if ``alt_dist``
            is not specified they are assumed to be drawn from the base
            distribution of the flow.
        alt_dist : :obj:`nflows.distribution.Distribution`
            Distribution object from which the latent samples z were
            drawn from. Must have a ``log_prob`` method that accepts an
            instance of ``torch.Tensor``

        Returns
        -------
        samples : ndarray
            Array containing samples in the latent space.
        log_prob : ndarray
            Array containing the log probability that corresponds to each
            sample.
        """
        if self.model is None:
            raise RuntimeError('Model is not initialised yet!')
        if self.model.training:
            self.model.eval()
        if z is None:
            with torch.no_grad():
                x, log_prob = self.model.sample_and_log_prob(int(N))
        else:
            if alt_dist is not None:
                log_prob_fn = alt_dist.log_prob
            else:
                log_prob_fn = self.model.base_distribution_log_prob

            with torch.no_grad():
                if isinstance(z, np.ndarray):
                    z = (
                        torch.from_numpy(z).type(torch.get_default_dtype())
                        .to(self.model.device)
                    )
                log_prob = log_prob_fn(z)
                x, log_J = self.model.inverse(z, context=None)
                log_prob -= log_J

        x = x.detach().cpu().numpy().astype(np.float64)
        log_prob = log_prob.detach().cpu().numpy().astype(np.float64)
        return x, log_prob

    def __getstate__(self):
        state = self.__dict__.copy()
        state['initialised'] = False
        del state['optimiser']
        del state['model']
        del state['model_config']
        return state


class CombinedFlowModel(FlowModel):
    """Flow Model that contains multiple flows for importance sampler."""

    models = torch.nn.ModuleList()

    def __init__(self, config=None, output='./'):
        super().__init__(config=config, output=output)
        self.weights_files = []

    @property
    def model(self):
        return self.models[-1]

    @model.setter
    def model(self, model):
        logger.info('In the setter')
        if model is not None:
            self.models.append(model)

    @property
    def n_models(self):
        return len(self.models)

    def initialise(self):
        """Initialise things"""
        self.initialised = True

    def reset_optimiser(self):
        """Reset the optimiser to point at current model."""
        self._optimiser = self.get_optimiser()

    def add_new_flow(self, reset=False):
        """Add a new flow"""
        logger.info('Add a new flow')
        if reset or not self.models:
            new_flow, self.device = configure_model(self.model_config)
        else:
            new_flow = copy.deepcopy(self.model)
        logger.debug(f'Training device: {self.device}')
        self.inference_device = torch.device(
            self.model_config.get('inference_device_tag', self.device)
            or self.device
        )
        logger.debug(f'Inference device: {self.inference_device}')
        self.models.eval()
        self.models.append(new_flow)
        self.reset_optimiser()

    def log_prob_ith(self, x, i):
        """Compute the log-prob for the ith flow"""
        x = (
            torch.from_numpy(x).type(torch.get_default_dtype())
            .to(self.model.device)
        )
        if self.models[i].training:
            self.models[i].eval()
        with torch.no_grad():
            log_prob = self.models[i].log_prob(x)
        log_prob = log_prob.cpu().numpy().astype(np.float64)
        return log_prob

    def log_prob_all(self, x, exclude_last=False):
        """Compute the log probability using all of the stored models."""
        x = (
            torch.from_numpy(x).type(torch.get_default_dtype())
            .to(self.model.device)
        )
        if self.models.training:
            self.models.eval()
        n = self.n_models
        if exclude_last:
            n -= 1
        log_prob = torch.empty(x.shape[0], n)
        with torch.no_grad():
            for i, m in enumerate(self.models[:n]):
                log_prob[:, i] = m.log_prob(x)
        log_prob = log_prob.cpu().numpy().astype(np.float64)
        return log_prob

    def sample_ith(self, i, N=1):
        """Draw samples from the ith flow"""
        if self.models is None:
            raise RuntimeError('Models are not initialised yet!')
        if self.models[i].training:
            self.models[i].eval()

        with torch.no_grad():
            x = self.models[i].sample(int(N))

        x = x.cpu().numpy().astype(np.float64)
        return x

    def save_weights(self, weights_file):
        """Save the weights file."""
        super().save_weights(weights_file)
        self.weights_files.append(self.weights_file)

    def load_all_weights(self):
        """Load all of the weights files for each flow.

        Resets any existing models.
        """
        if self.models:
            logger.debug('Resetting model list')
            self.models = torch.nn.ModuleList()
        logger.debug(f'Loading weights from {self.weights_files}')
        for wf in self.weights_files:
            new_flow, self.device = configure_model(self.model_config)
            new_flow.load_state_dict(torch.load(wf))
            self.models.append(new_flow)
        self.models.eval()

    def update_weights_path(self, weights_path):
        """Update the weights path.

        Searches in the specified directory for weights files.
        """
        all_weights_files = glob.glob(
            os.path.join(weights_path, '', 'level_*', 'model.pt')
        )
        if len(all_weights_files) != self.n_models:
            raise RuntimeError(
                f'Cannot use weights from: {weights_path}.'
            )
        self.weights_files = [
            os.path.join(weights_path, f'level_{i}', 'model.pt')
            for i in range(self.n_models)
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['initialised'] = False
        del state['optimiser']
        del state['model_config']
        return state
