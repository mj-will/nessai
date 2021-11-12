# -*- coding: utf-8 -*-
"""
Object and functions to handle training the normalising flow.
"""
import copy
import json
import logging
import numpy as np
import os
import shutil
import torch
from torch.nn.utils import clip_grad_norm_

from .flows import configure_model, reset_weights, reset_permutations
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
    def __init__(self, config=None, output='./'):
        self.model = None
        self.initialised = False
        self.output = output
        self.setup_from_input_dict(config)
        self.weights_file = None

        self.device = None
        self.inference_device = None
        self.use_dataloader = False

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

    def prep_data(self, samples, val_size, batch_size, use_dataloader=False):
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

        Returns
        -------
        train_data, val_data :
            Training and validation data as either a tensor or dataloader
        """
        if not self.initialised:
            self.initialise()

        idx = np.random.permutation(samples.shape[0])
        samples = samples[idx]

        logger.debug("N input samples: {}".format(len(samples)))

        # setup data loading
        n = int((1 - val_size) * samples.shape[0])
        x_train, x_val = samples[:n], samples[n:]
        logger.debug(f'{x_train.shape} training samples')
        logger.debug(f'{x_val.shape} validation samples')

        if not type(batch_size) is int:
            if batch_size == 'all':
                self.batch_size = x_train.shape[0]
            else:
                raise RuntimeError(f'Unknown batch size: {batch_size}')
        else:
            self.batch_size = batch_size
        dtype = torch.get_default_dtype()
        if use_dataloader:
            logger.debug('Using dataloaders')
            train_tensor = torch.from_numpy(x_train).type(dtype)
            train_dataset = torch.utils.data.TensorDataset(train_tensor)
            train_data = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True)

            val_tensor = torch.from_numpy(x_val).type(dtype)
            val_dataset = torch.utils.data.TensorDataset(val_tensor)
            val_data = torch.utils.data.DataLoader(
                val_dataset, batch_size=x_val.shape[0], shuffle=False)
        else:
            logger.debug('Using tensors')
            train_data = \
                torch.from_numpy(x_train).type(dtype).to(self.device)
            val_data = \
                torch.from_numpy(x_val).type(dtype).to(self.device)

        return train_data, val_data

    def _train(self, train_data, noise_scale=0.0, is_dataloader=False):
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
        else:
            def loss_fn(data):
                return -model.log_prob(data).mean()

        if not is_dataloader:
            p = torch.randperm(train_data.shape[0])
            train_data = train_data[p, :].split(self.batch_size)

        n = 0
        for data in train_data:
            if is_dataloader:
                data = data[0].to(self.device)
            if noise_scale:
                data += noise_scale * torch.randn_like(data)
            for param in model.parameters():
                param.grad = None
            loss = loss_fn(data)
            train_loss += loss.item()
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            self._optimiser.step()
            n += 1

        if self.annealing:
            self.scheduler.step()

        return train_loss / n

    def _validate(self, val_data, is_dataloader=False):
        """
        Loop over the data and get validation loss

        Parameters
        ----------
        val_data : :obj:`torch.util.data.Dataloader` or :obj:`torch.Tensor
            Dataloader with data to validate on
        is_dataloader : bool, optional
            Boolean to indicate if the data is a dataloader

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
        else:
            def loss_fn(data):
                return -model.log_prob(data).mean()

        if is_dataloader:
            n = 0
            for data in val_data:
                if is_dataloader:
                    data = data[0].to(self.device)
                with torch.no_grad():
                    val_loss += loss_fn(data).item()
                n += 1

            return val_loss / n
        else:
            with torch.no_grad():
                val_loss += loss_fn(val_data).item()
            return val_loss

    def train(self, samples, max_epochs=None, patience=None, output=None,
              val_size=None, plot=True):
        """
        Train the flow on a set of samples.

        Allows for training parameters to specified instead of those
        given in initial config.

        Parameters
        ----------
        samples : ndarray
            Unstructured numpy array containing data to train on
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

        train_data, val_data = self.prep_data(
            samples,
            val_size=val_size,
            batch_size=self.batch_size,
            use_dataloader=self.use_dataloader
        )

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

        if plot:
            history = dict(loss=[], val_loss=[])

        current_weights_file = output + 'model.pt'
        logger.debug(f'Training with {samples.shape[0]} samples')
        for epoch in range(1, max_epochs + 1):

            loss = self._train(
                train_data,
                noise_scale=noise_scale,
                is_dataloader=self.use_dataloader
            )
            val_loss = self._validate(
                val_data,
                is_dataloader=self.use_dataloader
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

        self.model.load_state_dict(best_model)
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
        if weights:
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

        z = z.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()
        return z, log_prob

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
                x, log_prob = self.model.sample_and_log_prob(N)
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

        x = x.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()
        return x, log_prob

    def __getstate__(self):
        state = self.__dict__.copy()
        state['initialised'] = False
        del state['optimiser']
        del state['model']
        del state['model_config']
        return state
