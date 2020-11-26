import copy
import json
import logging
import numpy as np
import os
import shutil
import torch
from torch.nn.utils import clip_grad_norm_

from .flows import setup_model, reset_weights, reset_permutations
from .plot import plot_loss
from .utils import FPJSONEncoder, compute_minimum_distances

logger = logging.getLogger(__name__)


def update_config(d):
    """
    Update the default configuration dictionary.

    The default configuration is:
        lr=0.001
        annealing=False
        batch_size=100
        val_size=0.1
        max_epochs=500
        patience=20
        noise_scale=0.0
        model_config=default_model

    where `default model` is:
        n_neurons=32
        n_blocks=4
        n_layers=2
        ftype='RealNVP'
        device_tag='cpu'
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
    default_model = dict(n_inputs=None, n_neurons=32, n_blocks=4, n_layers=2,
                         ftype='RealNVP', device_tag='cpu',
                         kwargs=dict(batch_norm_between_layers=True,
                                     linear_transform='lu'))

    default = dict(lr=0.001,
                   annealing=False,
                   clip_grad_norm=5,
                   batch_size=100,
                   val_size=0.1,
                   max_epochs=500,
                   patience=20,
                   noise_scale=0.0)

    if d is None:
        default['model_config'] = default_model
    else:
        if not isinstance(d, dict):
            raise TypeError('Must pass a dictionary to update the default '
                            'trainer settings')
        else:
            default.update(d)
            default_model.update(d.get('model_config', {}))
            default['model_config'] = default_model

    return default


class FlowModel:
    """
    Object that contains the normalsing flows and handles training and data
    pre-processing.

    Does NOT use stuctured arrays for live points, `Proposal` object
    should act as the interface between structured and unstructured arrays
    of live points.

    Parameters
    ----------
    config : dict, optional (None)
        Configuration used for the normalising flow. If None, default values
        are used.
    output : str, optional ('./')
        Path for output, this includes weights files and the loss plot.
    """
    def __init__(self, config=None, output='./'):
        self.model = None
        self.initialised = False
        self.output = output
        self.setup_from_input_dict(config)
        self.weights_file = None

    def save_input(self, config, output_file=None):
        """
        Save the dictionary used as an inputs as a JSON file in the output
        directory.

        Parameters
        ----------
        config : dict
            Dictionary to save
        ouput_file : str, optional
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
            json.dump(config, f, indent=4, cls=FPJSONEncoder)

    def setup_from_input_dict(self, config):
        """
        Setup the trainer from a dictionary, all keys in the dictionary are
        added as methods to the ocject. Input is automatically saved.

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
        """
        Get a the mask
        """
        pass

    def get_optimiser(self):
        """
        Get the optimiser and ensure it is always correctly intialised.

        Returns
        -------
        :obj:`torch.optim.Adam`
            Instance of the Adam optimiser from torch.optim
        """
        if self.model is None:
            raise RuntimeError('Cannot initialise optimiser before model')
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.lr, weight_decay=1e-6)

    def initialise(self):
        """
        Initialise the model and optimiser.

        This includes:
            * Updating the model configuration
            * Initialising the normalising flow
            * Initialiseing the optimiser
        """
        self.update_mask()
        self.model, self.device = setup_model(self.model_config)
        self.optimiser = self.get_optimiser()
        self.initialised = True

    def prep_data(self, samples, val_size, batch_size):
        """
        Prep data and return dataloaders for training
        """
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
                batch_size = x_train.shape[0]
            else:
                raise RuntimeError(f'Unknown batch size: {batch_size}')
        train_tensor = torch.from_numpy(x_train.astype(np.float32))
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        val_tensor = torch.from_numpy(x_val.astype(np.float32))
        val_dataset = torch.utils.data.TensorDataset(val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=x_val.shape[0],
                                                 shuffle=False)

        return train_loader, val_loader

    def _train(self, loader, noise_scale=0.0):
        """
        Loop over the data and update the weights

        Parameters
        ----------
        loader : :obj:`torch.util.data.Dataloader`
            Dataloader with data to train on
        noise_scale : float, optional (0.0)
            Scale of Gaussian noise added to data

        Returns
        -------
        Mean of training loss for each batch
        """
        model = self.model
        model.train()
        train_loss = 0

        if hasattr(model, 'loss_function'):
            loss_fn = model.loss_function
        else:
            def loss_fn(data):
                return -model.log_prob(data).mean()

        for idx, data in enumerate(loader):
            if noise_scale:
                data[0] += noise_scale * torch.randn_like(data[0])
            data = data[0].to(self.device)
            self.optimiser.zero_grad()
            loss = loss_fn(data)
            train_loss += loss.item()
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            self.optimiser.step()

        if self.annealing:
            self.scheduler.step()

        return train_loss / len(loader)

    def _validate(self, loader):
        """
        Loop over the data and get validation loss

        Parameters
        ----------
        loader : :obj:`torch.util.data.Dataloader`
            Dataloader with data to validate on

        Returns
        -------
        Mean of training loss for each batch
        """
        model = self.model
        model.eval()
        val_loss = 0

        if hasattr(model, 'loss_function'):
            loss_fn = model.loss_function
        else:
            def loss_fn(data):
                return -model.log_prob(data).mean()

        for idx, data in enumerate(loader):
            data = data[0].to(self.device)
            with torch.no_grad():
                val_loss += loss_fn(data).item()

        return val_loss / len(loader)

    def train(self, samples, max_epochs=None, patience=None, output=None,
              val_size=None, plot=True):
        """
        Train the flow on a set of samples.

        Allows for training parameters to specified instead of those
        given in initial config.

        Parameters
        ----------
        samples : :obj:`np.ndarray`
            Unstructured numpy array containing data to train on
        max_epochs : int, optional
            Maxinum number of epochs that is used instead of value
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
            noise_scale = 0.1 * np.std(compute_minimum_distances(samples))
            logger.debug(f'Using adaptive scale: {noise_scale:.3f}')
        else:
            noise_scale = self.noise_scale

        train_loader, val_loader = self.prep_data(samples, val_size=val_size,
                                                  batch_size=self.batch_size)

        if max_epochs is None:
            max_epochs = self.max_epochs
        if self.annealing:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimiser, max_epochs)
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

            loss = self._train(train_loader, noise_scale=noise_scale)
            val_loss = self._validate(val_loader)
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
        self.model.eval()

        if plot:
            plot_loss(epoch, history, output=output)

    def save_weights(self, weights_file):
        """
        Save the weights file. If the file already exists move it to
        `model.py.old` and then save the file.
        """
        if os.path.exists(weights_file):
            shutil.move(weights_file, weights_file + '.old')

        torch.save(self.model.state_dict(), weights_file)
        self.weights_file = weights_file

    def load_weights(self, weights_file):
        """
        Load weights for the model and initialiases the model if it is not
        intialised. The weights_file attribute is also updated.

        Model is loaded in evaluation mode (model.eval())

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
        Trys to the load the weights file and if not, trys to load
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
        Reset the weights of the model and optimiser
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
        self.optimiser = self.get_optimiser()
        logger.debug('Reseting optimiser')

    def forward_and_log_prob(self, x):
        """
        Forward pass through the model and return the samples in the latent
        space with their log probabilties

        Parameters
        ----------
        x : array_like
            Array of samples

        Returns
        -------
        z : :obj:`np.ndarray`
            Samples in the latent space
        log_prob : :obj:`np.ndarray`
            Log probabilties for each samples
        """
        x = torch.Tensor(x.astype(np.float32)).to(self.device)
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
        z : array_like, optional
            Array of latent samples to map the the data space, if `alt_dist`
            is not specified they are assumed to be drawn from the base
            distribution of the flow.
        alt_dist : :obj:`nflows.distribution.Distribution`
            Distribution object from which the latent samples z were
            drawn from. Must have a `log_prob` method that accepts an
            instance of torch.Tensor

        Returns
        -------
        samples : :obj:`np.ndarray`
            Tensor containing samples in the latent space
        log_prob : :obj:`np.ndarray`
            Tensor containing the log probabaility that corresponds to each
            sample
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
                    z = torch.Tensor(z.astype(np.float32)).to(self.device)
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
